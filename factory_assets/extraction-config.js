/**
 * lib/extraction-config.js — THE VALIDATOR AND THE INTERPRETER, BROWSER SIDE.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * READ THIS BEFORE CHANGING ANYTHING IN THIS FILE
 * ═══════════════════════════════════════════════════════════════════════════
 * This code runs in a content script on a third-party dealership page, inside an
 * extension that also holds the customer's live Facebook Marketplace session. The
 * "extraction config" it consumes was written by a language model whose only
 * input was that dealership's HTML — markup an attacker can edit.
 *
 * SO THE CONFIG IS DATA AND IS NEVER, EVER CODE. There is no eval, no Function,
 * no innerHTML, no setAttribute, no `javascript:` navigation and no DOM write of
 * any kind in this file. A config can name a CSS selector, an attribute, and one
 * of eight named readings that are implemented HERE, in advance, by us. It cannot
 * express an instruction, because there is no key in the schema whose value ever
 * becomes behaviour.
 *
 * It is validated on the server before it is sent AND again here before it is
 * used — including every time it comes back out of chrome.storage, because a
 * cache is not a trust boundary. lib/extraction-config.ts on the server is the
 * canonical copy of these rules; test/extraction-config.test.js runs one shared
 * corpus of hostile configs through BOTH and requires identical verdicts, because
 * a client that is more permissive than the server is a client with no gate.
 *
 * Exposes globalThis.AP_EXTRACTION — same pattern as AP_VEHICLE_MAPS, which is
 * how content scripts injected together in one isolated world share code.
 */

(() => {
    "use strict";

    const CONFIG_VERSION = 1;

    // ─────────────────────────── the vocabulary ───────────────────────────
    // Every one of these lists is an ALLOWLIST. An unknown key, field, reading or
    // source is a rejection — never an ignored extra. That is what makes
    // `__proto__`, `constructor`, `onclick` and every future variant of the same
    // idea fail without this file having heard of them.

    const FIELD_NAMES = [
        "detail_url", "vin", "stock", "year", "make", "model", "trim", "name",
        "price", "mileage", "distance_unit", "color_ext", "color_int",
        "transmission", "engine", "fuel", "body_type", "condition",
        "photo", "photos", "description", "mpg_city", "mpg_hwy",
    ];
    const FIELD_SET = new Set(FIELD_NAMES);
    const TRANSFORMS = new Set(["text", "int", "price", "year", "url", "vin", "unit", "condition"]);
    const DERIVE_SOURCES = new Set(["detail_url", "photo", "name", "card_text"]);
    const EXTRACTOR_KEYS = new Set(["sel", "attr", "from", "re", "as", "all", "max"]);
    const TOP_KEYS = new Set(["v", "origin", "card", "fields", "next", "notes"]);
    const BANNED_KEYS = new Set(["__proto__", "constructor", "prototype"]);

    const LIMITS = {
        rawBytes: 8192,
        selectors: 40,
        fields: 24,
        selectorChars: 200,
        attrChars: 40,
        regexChars: 120,
        regexes: 24,
        notesChars: 1000,
        photosMax: 20,
        regexProbeMs: 25,
        regexInputChars: 512,
        /** Whole-page apply budget. A config cannot spend the scan on one page. */
        applyBudgetMs: 4000,
        maxCards: 1200,
    };

    // ══════════════════════ the selector grammar ══════════════════════
    //
    // A LITERAL PORT of parseSelectorList in autoposter-web/lib/mini-dom.ts. It is
    // duplicated because there is no bundler between a Next server and an MV3
    // content script; the parity test is what keeps the two honest.
    //
    // Accepted:  tag  .class  #id  [attr]  [attr=v] [attr^=v] [attr$=v]
    //            [attr*=v] [attr~=v] [attr|=v] (optional trailing ` i`),
    //            descendant and child combinators, comma lists, `*`.
    // Rejected:  every pseudo-class and pseudo-element, `+`/`~` siblings,
    //            :nth-child, :has, :not, escapes, functional notation.
    //
    // The rejections carry weight beyond safety: `div > div:nth-child(3) > span`
    // is the signature of a selector written by counting boxes, and it breaks the
    // first time the dealer's vendor adds a badge row.

    const MAX_ALTERNATIVES = 4;
    const MAX_STEPS = 6;
    const IDENT = /^[A-Za-z_][A-Za-z0-9_-]*$/;
    const ATTR_COND_RE = /^\[\s*([a-zA-Z][-a-zA-Z0-9_:.]{0,40})\s*(?:([~^$*|]?=)\s*(?:"([^"<>]{0,120})"|'([^'<>]{0,120})'|([^\]\s"'<>]{1,120}))\s*(i)?\s*)?\]$/;

    function selErr(msg) { const e = new Error(msg); e.name = "SelectorError"; return e; }

    /** Blank out every [ ... ] section and quoted run, keeping the string's shape. */
    function stripBracketsAndQuotes(s) {
        let out = "";
        let depth = 0;
        let q = null;
        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (q) { out += " "; if (c === q) q = null; continue; }
            if (c === '"' || c === "'") { q = c; out += " "; continue; }
            if (c === "[") { depth++; out += " "; continue; }
            if (c === "]") { depth = Math.max(0, depth - 1); out += " "; continue; }
            out += depth > 0 ? " " : c;
        }
        return out;
    }

    function splitTop(s, sep) {
        const out = [];
        let depth = 0;
        let q = null;
        let start = 0;
        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (q) { if (c === q) q = null; continue; }
            if (c === '"' || c === "'") { q = c; continue; }
            if (c === "[") depth++;
            else if (c === "]") depth--;
            else if (c === sep && depth === 0) { out.push(s.slice(start, i)); start = i + 1; }
        }
        out.push(s.slice(start));
        return out.filter((x) => x.trim());
    }

    function parseCompound(s) {
        const out = { tag: null, id: null, classes: [], attrs: [] };
        let i = 0;
        let parts = 0;
        while (i < s.length) {
            parts++;
            if (parts > 12) throw selErr("compound selector has too many parts");
            const c = s[i];
            if (c === "*") {
                if (i !== 0) throw selErr("`*` may only start a compound");
                i++; continue;
            }
            if (c === "#" || c === ".") {
                let j = i + 1;
                while (j < s.length && !"#.[".includes(s[j])) j++;
                const name = s.slice(i + 1, j);
                if (!IDENT.test(name)) throw selErr(`illegal ${c === "#" ? "id" : "class"} name: ${name}`);
                if (c === "#") out.id = name; else out.classes.push(name);
                i = j; continue;
            }
            if (c === "[") {
                let depth = 0;
                let q = null;
                let j = i;
                for (; j < s.length; j++) {
                    const ch = s[j];
                    if (q) { if (ch === q) q = null; continue; }
                    if (ch === '"' || ch === "'") { q = ch; continue; }
                    if (ch === "[") depth++;
                    else if (ch === "]") { depth--; if (!depth) break; }
                }
                if (depth !== 0) throw selErr("unbalanced [ ] in selector");
                const raw = s.slice(i, j + 1);
                const m = ATTR_COND_RE.exec(raw);
                if (!m) throw selErr(`illegal attribute selector: ${raw}`);
                const name = m[1].toLowerCase();
                if (/^on/i.test(name)) throw selErr("event-handler attributes may not be selected on");
                out.attrs.push({ name, op: m[2] || null, value: m[3] !== undefined ? m[3] : (m[4] !== undefined ? m[4] : (m[5] !== undefined ? m[5] : null)), ci: Boolean(m[6]) });
                i = j + 1; continue;
            }
            let j = i;
            while (j < s.length && !"#.[".includes(s[j])) j++;
            const tag = s.slice(i, j);
            if (i !== 0) throw selErr(`unexpected "${tag}" in compound selector`);
            if (!/^[a-zA-Z][a-zA-Z0-9-]{0,20}$/.test(tag)) throw selErr(`illegal type selector: ${tag}`);
            out.tag = tag.toLowerCase();
            i = j;
        }
        if (!out.tag && !out.id && !out.classes.length && !out.attrs.length) throw selErr("empty compound selector");
        return out;
    }

    function parseComplex(s) {
        if (!s) throw selErr("empty selector alternative");
        const steps = [];
        let buf = "";
        let depth = 0;
        let q = null;
        const flush = () => {
            const t = buf.trim();
            buf = "";
            if (!t) return;
            steps.push(parseCompound(t));
        };
        for (let i = 0; i < s.length; i++) {
            const c = s[i];
            if (q) { buf += c; if (c === q) q = null; continue; }
            if (c === '"' || c === "'") { q = c; buf += c; continue; }
            if (c === "[") { depth++; buf += c; continue; }
            if (c === "]") { depth--; buf += c; continue; }
            if (depth === 0 && c === ">") { flush(); continue; }
            if (depth === 0 && /\s/.test(c)) { if (buf.trim()) flush(); continue; }
            buf += c;
        }
        flush();
        if (!steps.length) throw selErr("selector has no compound parts");
        if (steps.length > MAX_STEPS) throw selErr(`selector deeper than ${MAX_STEPS} steps`);
        return steps;
    }

    /**
     * Parse the accepted subset. Throws on anything outside it.
     *
     * Chrome's own querySelectorAll is what will EXECUTE the selector, so this
     * function's job is not to be a CSS engine — it is to be the gate that decides
     * which selectors Chrome is ever asked to run.
     */
    function parseSelectorList(input) {
        if (typeof input !== "string") throw selErr("selector must be a string");
        const s = input.trim();
        if (!s) throw selErr("empty selector");
        if (s.length > LIMITS.selectorChars) throw selErr(`selector longer than ${LIMITS.selectorChars} chars`);
        /* Structural rejections, run ONLY outside [ ] and quotes. Testing the raw
         * string rejected `script[type="application/ld+json"]` — the commonest
         * selector on a dealer page — because of the `+` in a MIME type. */
        const bare = stripBracketsAndQuotes(s);
        if (bare.includes(":")) throw selErr("pseudo-classes and pseudo-elements are not allowed");
        if (/[{}\\<]/.test(bare)) throw selErr("illegal character in selector");
        if (/[+~]/.test(bare)) throw selErr("sibling combinators (+ ~) are not allowed");
        const parts = splitTop(s, ",");
        if (parts.length > MAX_ALTERNATIVES) throw selErr(`at most ${MAX_ALTERNATIVES} comma alternatives`);
        return parts.map((p) => parseComplex(p.trim()));
    }

    // ══════════════════════ regex screening ══════════════════════

    const CONTROL_CHARS = new RegExp("[\\u0000-\\u001f\\u007f]");

    /**
     * REDOS SCREEN, STATIC HALF. Reject a quantifier applied to a group whose body
     * itself contains a quantifier or an alternation — (a+)+, (a|a)*, ([a-z]*)* —
     * which is the whole family of exponential-backtracking shapes. Broader than
     * the truly dangerous set, deliberately: a rejected regex costs the model one
     * retry, a hanging regex costs the customer their browser tab.
     */
    function staticRedosRisk(src) {
        for (let i = 0; i < src.length; i++) {
            if (src[i] !== ")") continue;
            const next = src[i + 1];
            const nested = next === "*" || next === "+" || next === "{" ||
                (next === "?" && (src[i + 2] === "*" || src[i + 2] === "+"));
            if (!nested) continue;
            let depth = 0;
            let open = -1;
            for (let j = i; j >= 0; j--) {
                if (src[j] === ")" && src[j - 1] !== "\\") depth++;
                else if (src[j] === "(" && src[j - 1] !== "\\") { depth--; if (!depth) { open = j; break; } }
            }
            if (open === -1) continue;
            const bare = src.slice(open + 1, i).replace(/\\./g, "");
            if (/[*+]/.test(bare) || /\{\d+,\d*\}/.test(bare) || bare.includes("|")) {
                return `nested quantifier over "(${src.slice(open + 1, i)})" — the classic catastrophic-backtracking shape`;
            }
        }
        if (/[*+]\s*[*+]/.test(src)) return "adjacent unbounded quantifiers";
        return null;
    }

    /**
     * REDOS SCREEN, EMPIRICAL HALF — a measurement of THIS regex rather than a
     * guess about its shape. Runs on the customer's machine, once per config, and
     * costs single-digit milliseconds for anything that passes.
     */
    function probeRegexCost(re, budgetMs) {
        const probes = [
            "a".repeat(400) + "!",
            "ab".repeat(200) + "!",
            "0".repeat(400) + "X",
            "$1,234 ".repeat(60),
            "A1B2C3D4E5F6G7H8J".repeat(24),
            " ".repeat(400) + "x",
        ];
        for (const p of probes) {
            const started = Date.now();
            try { re.lastIndex = 0; re.exec(p.slice(0, LIMITS.regexInputChars)); }
            catch { return "regex threw while executing"; }
            const spent = Date.now() - started;
            if (spent > budgetMs) return `regex took ${spent}ms on a ${LIMITS.regexInputChars}-char probe (budget ${budgetMs}ms)`;
        }
        return null;
    }

    function compileConfigRegex(src) {
        if (typeof src !== "string") return { ok: false, re: null, error: "re must be a string" };
        if (!src) return { ok: false, re: null, error: "re is empty" };
        if (src.length > LIMITS.regexChars) return { ok: false, re: null, error: `re longer than ${LIMITS.regexChars} chars` };
        if (CONTROL_CHARS.test(src)) return { ok: false, re: null, error: "re contains control characters" };
        if (!src.includes("(")) return { ok: false, re: null, error: "re must contain a capture group — it names what to keep" };
        const risk = staticRedosRisk(src);
        if (risk) return { ok: false, re: null, error: risk };
        let re;
        // `i` ONLY. A `g` regex carries lastIndex between calls, which silently
        // skips matches when one compiled object is reused across cards.
        try { re = new RegExp(src, "i"); }
        catch (e) { return { ok: false, re: null, error: `re does not compile: ${e.message}` }; }
        const cost = probeRegexCost(re, LIMITS.regexProbeMs);
        if (cost) return { ok: false, re: null, error: cost };
        return { ok: true, re, error: "" };
    }

    // ══════════════════════ the validator ══════════════════════

    function safeJsonParse(text) {
        let poisoned = "";
        try {
            const value = JSON.parse(text, function (key, val) {
                if (BANNED_KEYS.has(key)) { if (!poisoned) poisoned = key; return undefined; }
                return val;
            });
            /* STRIPPING IS NOT ENOUGH — IT HAS TO BE A REJECTION. The reviver
             * removes the key so nothing is polluted while we decide, and then
             * the document is refused: a config carrying `__proto__` is evidence
             * of a compromised model output or an injection through the dealer
             * page, and running the clean remainder throws that evidence away.
             * Five payloads in the attack corpus came back ok:true before this. */
            if (poisoned) return { ok: false, value: null, error: `config contains a prototype-poisoning key ("${poisoned}")` };
            return { ok: true, value, error: "" };
        } catch (e) {
            return { ok: false, value: null, error: `not valid JSON: ${e.message}` };
        }
    }

    const isPlainObject = (v) => typeof v === "object" && v !== null && !Array.isArray(v);

    function checkAttrName(name) {
        if (typeof name !== "string") return "attr must be a string";
        if (!name) return "attr is empty";
        if (name.length > LIMITS.attrChars) return `attr longer than ${LIMITS.attrChars} chars`;
        if (!/^[a-z][a-z0-9:_.-]*$/.test(name)) return `illegal attribute name "${name}"`;
        // Reading an inline handler returns a string, which is harmless in itself —
        // but that string is JavaScript source, and it would flow into a caption
        // and a published listing. No vehicle field lives in an on* attribute.
        if (/^on/.test(name)) return `event-handler attributes ("${name}") may not be read`;
        return null;
    }

    function checkOrigin(v) {
        if (typeof v !== "string" || !v) return "origin must be a string";
        if (v.length > 200) return "origin too long";
        let u;
        try { u = new URL(v); } catch { return "origin is not a URL"; }
        // Scheme ALLOWLIST on the parsed protocol.
        if (u.protocol !== "https:" && u.protocol !== "http:") return `origin scheme must be http(s), got ${u.protocol}`;
        if (u.origin !== v.replace(/\/+$/, "")) return "origin must be a bare scheme://host[:port]";
        return null;
    }

    /**
     * Validate a candidate config. Never throws; returns every reason it failed.
     * `expectOrigin` pins the config to the page it will run on — a config built
     * for a dealer site can never be pointed at facebook.com.
     */
    function validateConfig(raw, opts) {
        const errors = [];
        const warnings = [];
        let value = raw;

        if (typeof raw === "string") {
            if (raw.length > LIMITS.rawBytes) return { ok: false, config: null, errors: [`config is ${raw.length} bytes, limit ${LIMITS.rawBytes}`], warnings };
            const parsed = safeJsonParse(raw);
            if (!parsed.ok) return { ok: false, config: null, errors: [parsed.error], warnings };
            value = parsed.value;
        } else {
            let serialized = "";
            try { serialized = JSON.stringify(raw); } catch { return { ok: false, config: null, errors: ["config is not serialisable"], warnings }; }
            if (!serialized) return { ok: false, config: null, errors: ["config is empty"], warnings };
            if (serialized.length > LIMITS.rawBytes) return { ok: false, config: null, errors: [`config is ${serialized.length} bytes, limit ${LIMITS.rawBytes}`], warnings };
            // Re-parsed through the same reviver so an in-memory `__proto__` cannot
            // skip the stripping step.
            const reparsed = safeJsonParse(serialized);
            if (!reparsed.ok) return { ok: false, config: null, errors: [reparsed.error], warnings };
            value = reparsed.value;
        }

        if (!isPlainObject(value)) return { ok: false, config: null, errors: ["config must be a JSON object"], warnings };

        for (const k of Object.keys(value)) if (!TOP_KEYS.has(k)) errors.push(`unknown top-level key "${k}"`);
        if (value.v !== CONFIG_VERSION) errors.push(`v must be ${CONFIG_VERSION} (got ${JSON.stringify(value.v)})`);

        const originErr = checkOrigin(value.origin);
        if (originErr) errors.push(originErr);
        else if (opts && opts.expectOrigin && value.origin !== opts.expectOrigin) {
            errors.push(`origin "${String(value.origin).slice(0, 60)}" does not match this page ("${opts.expectOrigin}")`);
        }

        let selectorCount = 0;
        let regexCount = 0;
        const checkSelector = (s, where) => {
            if (typeof s !== "string") { errors.push(`${where}: selector must be a string`); return; }
            if (s.length > LIMITS.selectorChars) { errors.push(`${where}: selector longer than ${LIMITS.selectorChars} chars`); return; }
            selectorCount++;
            try { parseSelectorList(s); }
            catch (e) { errors.push(`${where}: ${e.message} — ${JSON.stringify(s).slice(0, 90)}`); }
        };

        checkSelector(value.card, "card");
        if (value.next !== undefined) checkSelector(value.next, "next");

        if (value.notes !== undefined) {
            if (typeof value.notes !== "string") errors.push("notes must be a string");
            else if (value.notes.length > LIMITS.notesChars) errors.push(`notes longer than ${LIMITS.notesChars} chars`);
        }

        if (!isPlainObject(value.fields)) {
            errors.push("fields must be an object");
            return { ok: false, config: null, errors, warnings };
        }
        const fieldKeys = Object.keys(value.fields);
        if (!fieldKeys.length) errors.push("fields is empty — a config that reads nothing is not a config");
        if (fieldKeys.length > LIMITS.fields) errors.push(`${fieldKeys.length} fields, limit ${LIMITS.fields}`);

        for (const name of fieldKeys) {
            if (!FIELD_SET.has(name)) { errors.push(`unknown field "${name}" — allowed: ${FIELD_NAMES.join(", ")}`); continue; }
            const ex = value.fields[name];
            if (!isPlainObject(ex)) { errors.push(`fields.${name} must be an object`); continue; }
            for (const k of Object.keys(ex)) if (!EXTRACTOR_KEYS.has(k)) errors.push(`fields.${name}: unknown key "${k}"`);
            /* NORMALISE THE EMPTY SPELLINGS BEFORE JUDGING THEM. An empty
             * `attr` means "read the text" and an empty `sel` means "the card
             * itself" — both already expressible by omitting the key, so this
             * only DELETES keys and can never admit a capability an empty string
             * did not have. Measured: three corpus dealerships lost all three
             * repair attempts to `"attr": ""`. KEPT BYTE-COMPATIBLE with
             * lib/extraction-config.ts — test/extraction-config.test.js requires
             * both sides to return the identical verdict for every case. */
            for (const k of ["sel", "attr", "re"]) {
                if (typeof ex[k] === "string" && ex[k] === "") delete ex[k];
            }
            const hasSel = ex.sel !== undefined;
            const hasFrom = ex.from !== undefined;
            if (hasSel && hasFrom) errors.push(`fields.${name}: use sel OR from, not both`);
            if (hasSel) checkSelector(ex.sel, `fields.${name}.sel`);
            if (hasFrom) {
                if (typeof ex.from !== "string" || !DERIVE_SOURCES.has(ex.from)) {
                    errors.push(`fields.${name}.from must be one of detail_url, photo, name, card_text`);
                } else if (ex.from === name) {
                    errors.push(`fields.${name}.from cannot be itself`);
                } else if (ex.attr !== undefined) {
                    errors.push(`fields.${name}: from and attr are mutually exclusive`);
                }
            }
            if (ex.attr !== undefined) {
                const e = checkAttrName(ex.attr);
                if (e) errors.push(`fields.${name}.${e}`);
            }
            if (ex.re !== undefined) {
                regexCount++;
                const r = compileConfigRegex(ex.re);
                if (!r.ok) errors.push(`fields.${name}.re: ${r.error}`);
            }
            if (ex.as !== undefined && (typeof ex.as !== "string" || !TRANSFORMS.has(ex.as))) {
                errors.push(`fields.${name}.as must be one of text, int, price, year, url, vin, unit, condition`);
            }
            if (ex.all !== undefined) {
                if (typeof ex.all !== "boolean") errors.push(`fields.${name}.all must be a boolean`);
                else if (ex.all && name !== "photos") errors.push(`fields.${name}: only "photos" may set all:true`);
            }
            if (ex.max !== undefined) {
                if (typeof ex.max !== "number" || !Number.isInteger(ex.max) || ex.max < 1 || ex.max > LIMITS.photosMax) {
                    errors.push(`fields.${name}.max must be an integer 1..${LIMITS.photosMax}`);
                }
            }
            if (name === "photos" && ex.all !== true) warnings.push('fields.photos without all:true reads only the first image');
        }

        if (selectorCount > LIMITS.selectors) errors.push(`${selectorCount} selectors, limit ${LIMITS.selectors}`);
        if (regexCount > LIMITS.regexes) errors.push(`${regexCount} regexes, limit ${LIMITS.regexes}`);
        if (!value.fields.detail_url && !value.fields.vin) {
            errors.push("a config must define detail_url or vin — without one there is no key to dedupe cars on");
        }

        if (errors.length) return { ok: false, config: null, errors, warnings };
        return { ok: true, config: value, errors: [], warnings };
    }

    // ══════════════════════ the transforms ══════════════════════
    // Ported from autoposter-web/lib/extraction-apply.ts. Identical output is a
    // TESTED property, not an intention: the server grades a config against these
    // semantics and certifies it, so a divergence here means the server certified
    // something the browser will not reproduce.

    const MAX_TEXT = 400;

    function tText(raw) {
        const s = raw.replace(/\s+/g, " ").trim().slice(0, MAX_TEXT);
        return s || null;
    }

    function tInt(raw) {
        const m = raw.replace(/[    ]/g, " ").match(/(\d{1,3}(?:[,\s]\d{3})+|\d+)/);
        if (!m) return null;
        const n = parseInt(m[1].replace(/[,\s]/g, ""), 10);
        return Number.isFinite(n) ? n : null;
    }

    /* A PRICE, not a number that happened to sit near a dollar sign. 0 is rejected
     * — it is what a scraper emits when it found the element and parsed the wrong
     * thing — and so is anything under 100, which is where a monthly payment and a
     * doc fee both land. Marhofer's card markup carries a $398 doc fee and a $50
     * filing fee in the same pricing block as the real $4,447. */
    const PAYMENT_MARKER = /\/\s*(mo|month|wk|week|bw)\b|\bper\s+(month|week)\b|\bbi-?weekly\b|\bmonthly\b|\bo\.?a\.?c\.?\b/i;

    function tPrice(raw) {
        /* A MONTHLY PAYMENT IS NOT A PRICE, AND MAGNITUDE CANNOT TELL THEM APART.
         * "$349/mo" and "$349" are the same integer; only the words beside it
         * differ, so the marker is read off the SAME text the number came from.
         * The set-level median bar in the server's grader catches the other half
         * of this trap (a fee row that is plausible per-value and absurd across
         * a whole lot). */
        if (PAYMENT_MARKER.test(raw)) return null;
        const n = tInt(raw);
        if (n === null) return null;
        if (n < 100 || n > 2000000) return null;
        return n;
    }

    function tYear(raw) {
        const m = raw.match(/\b(19\d{2}|20\d{2})\b/);
        if (!m) return null;
        const y = parseInt(m[1], 10);
        const max = new Date().getUTCFullYear() + 2;
        return y >= 1900 && y <= max ? y : null;
    }

    const CFG_VIN_RE = new RegExp("(?:^|[^A-HJ-NPR-Z0-9])([A-HJ-NPR-Z0-9]{17})(?:[^A-HJ-NPR-Z0-9]|$)", "i");

    function tVin(raw) {
        const m = raw.toUpperCase().match(CFG_VIN_RE);
        if (!m) return null;
        const vin = m[1];
        if (/^\d{17}$/.test(vin) || /^[A-Z]{17}$/.test(vin)) return null;
        return vin;
    }

    /* THE ODOMETER UNIT, READ AND NEVER GUESSED — the same rule content/dealersite.js
     * follows, and for the same reason: inferring it from the TLD published a
     * Canadian lot's whole inventory in miles at 1.6x the real distance. null is a
     * real answer; downstream falls back to the dealer's account setting. */
    function tUnit(raw) {
        if (/\b(km|kms|kilomet(?:er|re)s?|KMT)\b/i.test(raw)) return "km";
        if (/\b(mi|miles?|SMI)\b/i.test(raw)) return "mi";
        return null;
    }

    function tCondition(raw) {
        if (/\b(certified|pre-?owned|used|refurb)/i.test(raw)) return "used";
        if (/\bnew\b/i.test(raw)) return "new";
        return null;
    }

    /**
     * URL resolution. Scheme is an ALLOWLIST on the PARSED protocol, so
     * "java\nscript:alert(1)" — which every substring blocklist lets through,
     * because the browser strips the newline and the blocklist does not — fails
     * here. Credentials in the URL are refused outright.
     *
     * SAME-ORIGIN IS ENFORCED FOR PAGES AND NOT FOR PIXELS, and the split is the
     * point: detail_url and next are pages this scanner will FETCH, RENDER AND
     * PARSE with the customer's cookies, so a hostile card pointing them off-origin
     * would turn the scan into somebody else's crawler. A photo URL is data that
     * rides to our cloud as a string; it is never navigated to during a scan, and
     * dealer platforms serve every image from a CDN, so pinning it would break the
     * mechanism on the sites it exists for. WHICH RULE APPLIES IS CHOSEN BY THE
     * FIELD, NOT BY THE CONFIG — there is no way for a config to ask for
     * cross-origin on detail_url.
     */
    function resolveUrl(raw, base, allowCrossOrigin) {
        const s = String(raw).trim();
        if (!s || s.length > 2000) return null;
        let u;
        try { u = new URL(s, base); } catch { return null; }
        if (u.protocol !== "https:" && u.protocol !== "http:") return null;
        if (u.username || u.password) return null;
        if (!allowCrossOrigin) {
            let b;
            try { b = new URL(base); } catch { return null; }
            if (u.origin !== b.origin) return null;
        }
        return u.href;
    }

    const tUrl = (raw, base) => resolveUrl(raw, base, false);
    const tImageUrl = (raw, base) => resolveUrl(raw, base, true);

    function applyTransform(name, raw, base, field) {
        switch (name) {
            case "int": return tInt(raw);
            case "price": return tPrice(raw);
            case "year": return tYear(raw);
            case "url": return (field === "photo" || field === "photos") ? tImageUrl(raw, base) : tUrl(raw, base);
            case "vin": return tVin(raw);
            case "unit": return tUnit(raw);
            case "condition": return tCondition(raw);
            case "text":
            default: return tText(raw);
        }
    }

    // ══════════════════════ the interpreter ══════════════════════

    const DERIVED_FIRST = ["detail_url", "photo", "name"];

    /**
     * Run a VALIDATED config against a real document.
     *
     * THE ONLY DOM CALLS IN THIS FUNCTION ARE querySelector, querySelectorAll,
     * getAttribute AND textContent. Nothing is written. Nothing is evaluated. The
     * config's strings are passed to querySelector — which cannot execute code —
     * and to getAttribute, which returns one. That is the whole mechanism, and
     * test/extraction-config.test.js greps this file for every banned construct so
     * it stays that way.
     *
     * `doc` may be the live document or a DOMParser result (inert: no scripts run,
     * no resources load) — the interpreter cannot tell and must not care.
     */
    function applyConfig(cfg, doc, pageUrl, opts) {
        const started = Date.now();
        const budget = (opts && opts.budgetMs) || LIMITS.applyBudgetMs;
        const out = [];
        let cards = [];
        try { cards = Array.prototype.slice.call(doc.querySelectorAll(cfg.card), 0, LIMITS.maxCards); }
        catch { cards = []; }

        const compiled = new Map();
        for (const name of Object.keys(cfg.fields)) {
            const ex = cfg.fields[name];
            if (!ex || ex.re === undefined) continue;
            const r = compileConfigRegex(ex.re);
            if (r.ok) compiled.set(name, r.re);
        }

        const runRe = (name, raw) => {
            const re = compiled.get(name);
            if (!re) return raw;
            // Length-capped BEFORE the regex sees it: the third independent bound
            // on how much time one config can burn, after the static screen and
            // the empirical probe.
            const m = re.exec(String(raw).slice(0, LIMITS.regexInputChars));
            if (!m) return null;
            return m[1] !== undefined ? m[1] : m[0];
        };

        const names = Object.keys(cfg.fields);
        const ordered = DERIVED_FIRST.filter((n) => names.indexOf(n) !== -1)
            .concat(names.filter((n) => DERIVED_FIRST.indexOf(n) === -1));

        let budgetHit = false;
        for (const card of cards) {
            if (Date.now() - started > budget) { budgetHit = true; break; }
            const rec = Object.create(null);
            for (const name of ordered) {
                const ex = cfg.fields[name];
                if (!ex) continue;
                try {
                    if (ex.from) {
                        const src = ex.from === "card_text"
                            ? (card.textContent || "")
                            : (rec[ex.from] == null ? "" : String(rec[ex.from]));
                        if (!src) { rec[name] = null; continue; }
                        const piece = runRe(name, src);
                        rec[name] = piece == null ? null : applyTransform(ex.as, piece, pageUrl, name);
                        continue;
                    }
                    if (name === "photos" && ex.all) {
                        const list = [];
                        const nodes = ex.sel ? card.querySelectorAll(ex.sel) : [card];
                        const cap = Math.min(ex.max || LIMITS.photosMax, LIMITS.photosMax);
                        for (const n of nodes) {
                            if (list.length >= cap) break;
                            const raw = ex.attr ? n.getAttribute(ex.attr) : (n.textContent || "");
                            if (!raw) continue;
                            const piece = runRe(name, raw);
                            if (piece == null) continue;
                            const url = tImageUrl(piece, pageUrl);
                            if (url && list.indexOf(url) === -1) list.push(url);
                        }
                        rec.photos = list.length ? list : null;
                        continue;
                    }
                    let node = card;
                    if (ex.sel) {
                        node = card.querySelector(ex.sel);
                        if (!node) { rec[name] = null; continue; }
                    }
                    const raw = ex.attr ? node.getAttribute(ex.attr) : (node.textContent || "");
                    if (raw == null) { rec[name] = null; continue; }
                    const piece = runRe(name, raw);
                    rec[name] = piece == null ? null : applyTransform(ex.as, piece, pageUrl, name);
                } catch {
                    // One bad field never costs the whole card.
                    rec[name] = null;
                }
            }
            // A card with no key is not a car — it is a filter chip, a nav item or
            // a promo tile that happened to match the container selector.
            if (!rec.detail_url && !rec.vin) continue;
            // Copied onto a normal object so downstream code that expects a plain
            // record (JSON.stringify, spread) behaves; the null-prototype above is
            // what kept `__proto__`-named DOM data from ever mattering.
            out.push(Object.assign({}, rec));
        }

        return { vehicles: out, cards: cards.length, budgetHit };
    }

    globalThis.AP_EXTRACTION = {
        CONFIG_VERSION,
        LIMITS,
        FIELD_NAMES,
        validateConfig,
        parseSelectorList,
        compileConfigRegex,
        applyConfig,
        transforms: { tText, tInt, tPrice, tYear, tVin, tUnit, tCondition, tUrl, tImageUrl },
    };
})();
