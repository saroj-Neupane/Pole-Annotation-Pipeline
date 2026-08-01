/* Shared span-annotation viz: THE single chip/marker rendering style.
 *
 * Used by the demo page (interactive pan/zoom span viewer, GT overlay) and the
 * landing page (static showcase strip). Anything drawn over a span photo —
 * palette, chip geometry, marker builders, the projective ruler height fit —
 * lives here so the two pages can never drift apart.
 *
 * Marker contract: { x, y (percent), label, labelColor, heightLabel,
 *   heightBg?, heightFg?, anchored?, child?, gt?, isPoleTop?, traceIndex? }.
 */
(function (global) {
    'use strict';

    var NS = 'http://www.w3.org/2000/svg';

    // One dedicated hue per cable type, shared by the wedge and its catenary.
    // Palette is spread around the hue wheel so no two classes look alike;
    // red/maroon are RESERVED for failing calibration and never used here.
    var CABLE_COLORS = {
        primary: '#7c3aed',        // violet
        neutral: '#f97316',        // orange
        secondary: '#2563eb',      // blue
        open_secondary: '#0ea5e9', // sky
        catv: '#84cc16',           // lime
        telco: '#0d9488',          // teal
        fiber: '#16a34a',          // green
        comm: '#06b6d4',           // cyan (coarse fallback)
        guy: '#64748b', down_guy: '#64748b',   // slate
        unspecified: '#123A6D'     // navy
    };
    var CABLE_NAMES = {
        primary: 'Primary', neutral: 'Neutral', secondary: 'Secondary',
        open_secondary: 'Open Secondary', catv: 'CATV', telco: 'Telco',
        fiber: 'Fiber', comm: 'Comm', guy: 'Guy', down_guy: 'Down Guy',
        unspecified: 'Unspecified'
    };
    // Equipment hues distinct from every cable hue (and from each other).
    var EQUIPMENT_COLORS = {
        transformer: '#d946ef',         // magenta
        street_light: '#f59e0b',        // amber
        riser: '#155e75',               // dark cyan
        secondary_drip_loop: '#ec4899'  // pink
    };
    var EQUIPMENT_FALLBACK = '#475569';   // slate — no maroon (red reserved)
    var ARM_COLOR = '#8d6e63';
    var HEIGHT_WHITE = '#ffffff';
    var CALIB_GREEN = '#16a34a';
    var CALIB_YELLOW = '#ca8a04';   // tick disagrees slightly with the fit
    var CALIB_RED = '#dc2626';      // tick seriously off the fit
    var TICK_PASS_IN = 2.0, TICK_WARN_IN = 6.0;  // leave-one-out thresholds (in)
    var POLE_TOP_BLUE = '#123A6D';
    var MARKER_OPACITY = 0.70;      // photo stays readable underneath

    // ------------------------------------------------------------------
    // Text + formatting helpers
    // ------------------------------------------------------------------
    function measureText(label, fontPx) {
        var ctx = measureText._ctx || (measureText._ctx = document.createElement('canvas').getContext('2d'));
        ctx.font = '600 ' + fontPx + 'px system-ui, -apple-system, "Segoe UI", sans-serif';
        return ctx.measureText(label).width;
    }
    function titleCase(s) {
        s = String(s == null ? '' : s).trim();
        return s.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').replace(/\b\w/g, function (m) { return m.toUpperCase(); });
    }
    function cableName(t) { return CABLE_NAMES[t] || titleCase(t); }
    function cableColor(t) { return CABLE_COLORS[t] || CABLE_COLORS.unspecified; }
    function formatFeetInches(v) {
        var n = parseFloat(v);
        if (!isFinite(n)) return String(v == null ? '' : v);
        var ft = Math.floor(n), inch = Math.round((n - ft) * 12);
        if (inch >= 12) { ft += 1; inch = 0; }
        return ft + "'-" + inch + '"';
    }

    // ------------------------------------------------------------------
    // Projective ruler height model, same form the backend fits:
    //   inches = (a + b*x) / (1 + c*x),  x = percentY / 100
    // linearised to h = a + b*x - c*x*h and solved by normal equations.
    // ------------------------------------------------------------------
    function solve3x3(A, rhs) {
        var m = [[A[0][0], A[0][1], A[0][2], rhs[0]],
                 [A[1][0], A[1][1], A[1][2], rhs[1]],
                 [A[2][0], A[2][1], A[2][2], rhs[2]]];
        for (var c = 0; c < 3; c++) {
            var p = c;
            for (var r = c + 1; r < 3; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
            if (Math.abs(m[p][c]) < 1e-12) return null;
            var t = m[c]; m[c] = m[p]; m[p] = t;
            for (var r2 = 0; r2 < 3; r2++) {
                if (r2 === c) continue;
                var f = m[r2][c] / m[c][c];
                for (var k = c; k < 4; k++) m[r2][k] -= f * m[c][k];
            }
        }
        return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
    }

    // Raw projective fit (no sanity rejection): y_pct -> feet, or null.
    function fitProjection(ticks) {
        var pts = (ticks || []).filter(function (t) {
            return isFinite(+t.percentY) && isFinite(+t.height);
        });
        var uniq = {};
        pts.forEach(function (t) { uniq[(+t.percentY).toFixed(4)] = 1; });
        if (Object.keys(uniq).length < 3) return null;

        var A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], rhs = [0, 0, 0];
        pts.forEach(function (t) {
            var x = +t.percentY / 100, hIn = +t.height * 12;
            var row = [1, x, -x * hIn];
            for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 3; j++) A[i][j] += row[i] * row[j];
                rhs[i] += row[i] * hIn;
            }
        });
        var sol = solve3x3(A, rhs);
        if (!sol) return null;
        var a = sol[0], b = sol[1], c = sol[2];
        return function (yPct) {
            var x = +yPct / 100, den = 1 + c * x;
            if (!isFinite(den) || Math.abs(den) < 1e-9) return null;
            var inches = (a + b * x) / den;
            return inches > 0 ? inches / 12 : null;
        };
    }

    // Fit + sanity: reject a fit that can't reproduce its own anchors within 1 ft.
    function fitHeightModel(ticks) {
        var fn = fitProjection(ticks);
        if (!fn) return null;
        var pts = (ticks || []).filter(function (t) {
            return isFinite(+t.percentY) && isFinite(+t.height);
        });
        for (var k = 0; k < pts.length; k++) {
            var got = fn(pts[k].percentY);
            if (got == null || Math.abs(got - +pts[k].height) > 1.0) return null;
        }
        return fn;
    }

    // Leave-one-out check of one tick against the projection model: refit on
    // the OTHER ticks, predict this one, error in inches (null = unverifiable).
    function tickErrorInches(ticks, i) {
        if (!isFinite(+ticks[i].percentY) || !isFinite(+ticks[i].height)) return null;
        var others = ticks.filter(function (_, j) { return j !== i; });
        var fn = fitProjection(others) || fitProjection(ticks);
        if (!fn) return null;
        var got = fn(ticks[i].percentY);
        return got == null ? null : Math.abs(got - +ticks[i].height) * 12;
    }

    // ------------------------------------------------------------------
    // Marker builders (percent coords in, marker contract out)
    // ------------------------------------------------------------------
    function calibrationMarkers(results, heightAt) {
        var out = [];
        var ticks = (results || {})['Ruler Markings'] || [];
        ticks.forEach(function (kp, i) {
            // ruler ticks: wedge tip on the tick itself; colour = agreement
            // with the projective height model fitted on the other ticks
            var err = tickErrorInches(ticks, i);
            var color = CALIB_GREEN;
            if (err != null && err > TICK_WARN_IN) color = CALIB_RED;
            else if (err != null && err > TICK_PASS_IN) color = CALIB_YELLOW;
            out.push({
                x: +kp.percentX, y: +kp.percentY, heightLabel: formatFeetInches(kp.height),
                label: '', labelColor: color, heightBg: color,
                heightFg: '#fff', anchored: true
            });
        });
        var pt = (results || {})['Pole Top'];
        if (pt) {
            var h = heightAt ? heightAt(pt.percentY) : null;
            out.push({
                x: +pt.percentX, y: +pt.percentY,
                heightLabel: h == null ? '' : formatFeetInches(h),
                label: 'Pole Top', labelColor: POLE_TOP_BLUE, isPoleTop: true
            });
        }
        return out;
    }

    function equipmentMarkers(results, heightAt) {
        var out = [];
        ((results || {})['Equipment'] || []).forEach(function (det) {
            var typeLabel = titleCase(det.type);
            var color = EQUIPMENT_COLORS[det.type] || EQUIPMENT_FALLBACK;
            (det.keypoints || []).forEach(function (kp) {
                var kpLabel = titleCase(kp.name);
                var h = heightAt ? heightAt(kp.percentY) : null;
                out.push({
                    x: +kp.percentX, y: +kp.percentY,
                    heightLabel: h == null ? '' : formatFeetInches(h),
                    label: kpLabel && kpLabel !== typeLabel ? typeLabel + ' ' + kpLabel : (kpLabel || typeLabel),
                    labelColor: color
                });
            });
        });
        return out;
    }

    // Per-trace style: colour from the cable type; every catenary is SOLID.
    function buildTraceStyles(payload) {
        var styles = [];
        (payload && payload.traces || []).forEach(function (tr, i) {
            var ct = tr.cable_type || 'unspecified';
            styles[i] = { cableType: ct, color: cableColor(ct) };
        });
        return styles;
    }

    // Span-trace markers for one photo of the /demo/trace payload.
    // styleOf(traceIndex) -> {cableType, color} (see buildTraceStyles).
    function tracingMarkers(pd, traces, styleOf) {
        var out = [];
        if (!pd) return out;
        if (pd.role === 'midspan') {
            // matcher output only: dustbinned (unmatched) detections are hidden,
            // including wires whose trace reaches NEITHER pole attachment
            (pd.wires || []).forEach(function (w) {
                if (w.trace_index == null) return;
                var tr = traces[w.trace_index];
                if (!tr || (!tr.pole_a_attachment && !tr.pole_b_attachment)) return;
                var st = styleOf(w.trace_index);
                out.push({
                    x: +w.x, y: +w.y, heightLabel: w.height_label || '',
                    label: cableName(st.cableType),
                    labelColor: st.color, traceIndex: w.trace_index
                });
            });
        } else {
            (pd.attachments || []).forEach(function (att) {
                var isArm = (att.wire_count || 1) > 1;
                var insulator = att.insulator_name || titleCase(att.hardware || '');
                // fine class keeps fiber/telco/catv distinct; coarse hint says 'comm'
                var ct = att.role === 'guying' ? 'guy'
                       : (att.cable_type_fine || att.cable_type_hint || '');
                if (isArm) {
                    // Crossarm: the ARM is the pole-side object; K wires nest under it
                    out.push({
                        x: +att.x, y: +att.y, heightLabel: att.height_label || '',
                        label: 'Crossarm ×' + att.wire_count,
                        labelColor: ARM_COLOR, traceIndex: null, arm: true
                    });
                    (att.trace_indices || []).forEach(function (ti) {
                        var st = styleOf(ti);
                        out.push({
                            x: +att.x, y: +att.y, heightLabel: '',
                            label: insulator + ' · ' + cableName(st.cableType),
                            labelColor: st.color, traceIndex: ti, child: true
                        });
                    });
                    return;
                }
                var ctName = ct ? cableName(ct) : '';
                // guys carry no insulator — their name already IS the type
                var redundant = !ctName || !insulator ||
                                insulator.toLowerCase().indexOf(ctName.toLowerCase()) !== -1;
                var label = redundant ? (insulator || ctName) : insulator + ' · ' + ctName;
                var ti = (att.trace_indices && att.trace_indices.length) ? att.trace_indices[0] : null;
                out.push({
                    x: +att.x, y: +att.y, heightLabel: att.height_label || '',
                    label: label, labelColor: cableColor(ct || 'unspecified'),
                    traceIndex: ti, traceIndices: att.trace_indices || []
                });
            });
            if (pd.pole_top) {
                out.push({
                    x: +pd.pole_top.x, y: +pd.pole_top.y,
                    heightLabel: pd.pole_top.height_label || '',
                    label: 'Pole Top', labelColor: POLE_TOP_BLUE, isPoleTop: true
                });
            }
        }
        return out;
    }

    // Ground-truth markers (from /api/spans/<id>/gt — local label store only).
    function gtColor(base) {
        if (base === 'pole_top') return POLE_TOP_BLUE;
        return CABLE_COLORS[base] || EQUIPMENT_COLORS[base] || EQUIPMENT_FALLBACK;
    }
    function gtMarkers(gtPhoto) {
        var out = [];
        if (!gtPhoto) return out;
        function push(pt, base, label) {
            out.push({
                x: +pt.x, y: +pt.y, heightLabel: pt.height_label || '',
                label: label, labelColor: gtColor(base), gt: true, traceIndex: null
            });
        }
        (gtPhoto.attachments || []).forEach(function (a) {
            var m = /^([a-z_]+?)_?(\d*)$/i.exec(a.name || '') || [null, a.name || '', ''];
            push(a, m[1], titleCase(m[1]) + (m[2] ? ' ' + m[2] : ''));
        });
        (gtPhoto.wires || []).forEach(function (w, i) {
            // midspan GT carries the Katapult trace cable type when resolvable
            var ct = w.cable_type || null;
            push(w, ct || 'unspecified',
                 (ct ? cableName(ct) : 'Wire') + ' ' + (i + 1));
        });
        if (gtPhoto.pole_top) push(gtPhoto.pole_top, 'pole_top', 'Pole Top');
        return out;
    }

    // ------------------------------------------------------------------
    // Chip renderer primitives (image-pixel space)
    // ------------------------------------------------------------------

    // Stack rows without overlap inside [margin, natH-margin]; anchored rows
    // hang ABOVE their point (chip bottom = pointer tip on the keypoint).
    function stackRows(list, anchorBottom, geom) {
        var margin = geom.margin, bodyH = geom.bodyH, minStep = geom.minStep, natH = geom.natH;
        var prevTop = null;
        list.forEach(function (it) {
            var want = anchorBottom ? it.py - bodyH : it.py - bodyH / 2;
            var top = Math.max(margin, Math.min(natH - margin - bodyH, want));
            if (prevTop != null && top < prevTop + minStep) top = prevTop + minStep;
            it.top = top;
            prevTop = top;
        });
        var overflow = list.length ? list[list.length - 1].top + bodyH + margin - natH : 0;
        if (overflow > 0) {
            for (var i = list.length - 1; i >= 0; i--) {
                var minTop = i > 0 ? list[i - 1].top + minStep : margin;
                var shifted = Math.max(minTop, list[i].top - overflow);
                overflow -= (list[i].top - shifted);
                list[i].top = shifted;
            }
        }
    }

    // Measure markers into row items (px/py in image px, chip widths).
    function measureMarkers(markers, natW, natH, geom) {
        return markers.map(function (m) {
            var hW = m.heightLabel ? measureText(m.heightLabel, geom.fontPx) + 2 * geom.padX : 0;
            var lW = m.label ? measureText(m.label, geom.fontPx) + 2 * geom.padX : 0;
            return Object.assign({}, m, {
                px: m.x * natW / 100, py: m.y * natH / 100, hW: hW, lW: lW,
                totW: hW + lW + (hW && lW ? geom.chipGap : 0)
            });
        }).sort(function (a, b) { return a.py - b.py || a.px - b.px; });
    }

    // One chip row: rectangular chips + ONE colored pointer triangle on the
    // near edge + a thin leader from the triangle apex to the true keypoint.
    // side: 'right' = chips right of point (pointer on the left edge),
    //       'left'  = chips left of point (pointer on the right edge).
    // opts: { noPointer, popDelay } — popDelay animates the row in (ms).
    function drawChipRow(svg, it, chipLeft, geom, side, tipX, tipY, opts) {
        opts = opts || {};
        var g = document.createElementNS(NS, 'g');
        if (opts.popDelay != null) {
            g.setAttribute('class', 'marker-pop');
            g.style.animationDelay = opts.popDelay + 'ms';
        }
        var color = it.labelColor || POLE_TOP_BLUE;
        var bodyH = geom.bodyH, fontPx = geom.fontPx, ptrW = geom.ptrW, chipGap = geom.chipGap;
        var top = it.top, bot = it.top + bodyH, mid = it.top + bodyH / 2;
        var s = ptrW / 8;   // ptrW is 8 screen px scaled to image px

        var chips = [];
        if (it.hW) chips.push({ w: it.hW, bg: it.heightBg || HEIGHT_WHITE,
                                fg: it.heightFg || '#111', text: it.heightLabel });
        if (it.lW) chips.push({ w: it.lW, bg: color, fg: '#fff', text: it.label });
        if (!chips.length) return g;

        var totW = chips.reduce(function (a, c) { return a + c.w; }, 0) +
                   chipGap * (chips.length - 1);
        var nearEdge = side === 'right' ? chipLeft : chipLeft + totW;
        var apexX = side === 'right' ? nearEdge - ptrW : nearEdge + ptrW;

        if (!opts.noPointer) {
            // leader: crisp line from the keypoint to the pointer apex (skipped
            // when the apex already sits on the point, e.g. unstacked ticks)
            if (Math.abs(tipX - apexX) > 0.5 || Math.abs(tipY - bot) > 0.5) {
                var lead = document.createElementNS(NS, 'line');
                lead.setAttribute('x1', String(tipX)); lead.setAttribute('y1', String(tipY));
                lead.setAttribute('x2', String(apexX)); lead.setAttribute('y2', String(bot));
                lead.setAttribute('stroke', color);
                lead.setAttribute('stroke-width', String(1.3 * s));
                lead.setAttribute('stroke-linecap', 'round');
                g.appendChild(lead);
            }
            // pointer: fixed-size colored triangle, tip at the chip BOTTOM corner
            var tri = document.createElementNS(NS, 'path');
            tri.setAttribute('d', 'M ' + apexX + ' ' + bot +
                                  ' L ' + nearEdge + ' ' + top +
                                  ' L ' + nearEdge + ' ' + bot + ' Z');
            tri.setAttribute('fill', color);
            tri.setAttribute('fill-opacity', String(MARKER_OPACITY));
            g.appendChild(tri);
        }

        var cx = chipLeft;
        chips.forEach(function (c) {
            var r = document.createElementNS(NS, 'rect');
            r.setAttribute('x', String(cx)); r.setAttribute('y', String(top));
            r.setAttribute('width', String(c.w)); r.setAttribute('height', String(bodyH));
            r.setAttribute('fill', c.bg);
            r.setAttribute('fill-opacity', String(MARKER_OPACITY));
            g.appendChild(r);
            var t = document.createElementNS(NS, 'text');
            t.setAttribute('x', String(cx + c.w / 2));
            t.setAttribute('y', String(mid + fontPx * 0.05));
            t.setAttribute('text-anchor', 'middle');
            t.setAttribute('dominant-baseline', 'middle');
            t.setAttribute('fill', c.fg);
            t.setAttribute('font-size', String(fontPx));
            t.setAttribute('font-weight', '600');
            t.setAttribute('font-family', 'system-ui, -apple-system, Segoe UI, sans-serif');
            t.textContent = c.text;
            g.appendChild(t);
            cx += c.w + chipGap;
        });
        svg.appendChild(g);
        return g;
    }

    // Encloses a crossarm's nested wire rows in one bracket, so the K
    // conductors read as belonging to the arm above them.
    function drawArmBox(svg, run, bodyH, s) {
        if (!run.length) return;
        var pad = 3 * s;
        var x0 = Infinity, x1 = -Infinity;
        run.forEach(function (it) {
            x0 = Math.min(x0, it.chipLeft);
            x1 = Math.max(x1, it.chipLeft + it.totW);
        });
        var y0 = run[0].top - pad;
        var y1 = run[run.length - 1].top + bodyH + pad;
        var box = document.createElementNS(NS, 'rect');
        box.setAttribute('x', String(x0 - pad));
        box.setAttribute('y', String(y0));
        box.setAttribute('width', String(x1 - x0 + 2 * pad));
        box.setAttribute('height', String(y1 - y0));
        box.setAttribute('rx', String(3 * s));
        box.setAttribute('fill', ARM_COLOR);
        box.setAttribute('fill-opacity', '0.28');
        box.setAttribute('stroke', ARM_COLOR);
        box.setAttribute('stroke-width', String(1.4 * s));
        box.setAttribute('stroke-opacity', '0.85');
        svg.appendChild(box);
    }

    // Yellow ruler guide: least-squares line through the calibration tick
    // anchors, extended from the top of the ruler to the top of the photo.
    function drawRulerGuide(svg, tickPts, strokeW, cls) {
        if (!tickPts || tickPts.length < 2) return;
        var n = tickPts.length, sy = 0, sx = 0, syy = 0, sxy = 0;
        tickPts.forEach(function (p) { sy += p.y; sx += p.x; syy += p.y * p.y; sxy += p.x * p.y; });
        var den = n * syy - sy * sy;
        if (Math.abs(den) <= 1e-6) return;
        var slope = (n * sxy - sy * sx) / den;   // x per y (ruler ~vertical)
        var x0 = (sx - slope * sy) / n;
        var yTopTick = Math.min.apply(null, tickPts.map(function (p) { return p.y; }));
        var guide = document.createElementNS(NS, 'line');
        guide.setAttribute('x1', String(x0 + slope * yTopTick));
        guide.setAttribute('y1', String(yTopTick));
        guide.setAttribute('x2', String(x0));
        guide.setAttribute('y2', '0');
        guide.setAttribute('stroke', '#facc15');
        guide.setAttribute('stroke-width', String(strokeW));
        guide.setAttribute('stroke-linecap', 'round');
        guide.setAttribute('opacity', '0.95');
        if (cls) guide.setAttribute('class', cls);
        svg.appendChild(guide);
    }

    global.SpanViz = {
        NS: NS,
        CABLE_COLORS: CABLE_COLORS, CABLE_NAMES: CABLE_NAMES,
        EQUIPMENT_COLORS: EQUIPMENT_COLORS, EQUIPMENT_FALLBACK: EQUIPMENT_FALLBACK,
        ARM_COLOR: ARM_COLOR, HEIGHT_WHITE: HEIGHT_WHITE,
        CALIB_GREEN: CALIB_GREEN, CALIB_YELLOW: CALIB_YELLOW, CALIB_RED: CALIB_RED,
        TICK_PASS_IN: TICK_PASS_IN, TICK_WARN_IN: TICK_WARN_IN,
        POLE_TOP_BLUE: POLE_TOP_BLUE, MARKER_OPACITY: MARKER_OPACITY,
        measureText: measureText, titleCase: titleCase,
        cableName: cableName, cableColor: cableColor, formatFeetInches: formatFeetInches,
        fitProjection: fitProjection, fitHeightModel: fitHeightModel,
        tickErrorInches: tickErrorInches,
        calibrationMarkers: calibrationMarkers, equipmentMarkers: equipmentMarkers,
        buildTraceStyles: buildTraceStyles, tracingMarkers: tracingMarkers,
        gtColor: gtColor, gtMarkers: gtMarkers,
        measureMarkers: measureMarkers, stackRows: stackRows,
        drawChipRow: drawChipRow, drawArmBox: drawArmBox, drawRulerGuide: drawRulerGuide
    };
})(window);
