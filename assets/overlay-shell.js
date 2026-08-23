(function () {
  'use strict';

  var form = document.getElementById('overlay-form');
  var input = document.getElementById('overlay-url');
  var stage = document.getElementById('preview-stage');
  var scroll = document.getElementById('preview-scroll');
  var canvas = document.getElementById('preview-canvas');
  var image = document.getElementById('preview-image');
  var outline = document.getElementById('target-outline');
  var captureState = document.getElementById('capture-state');
  var status = document.getElementById('overlay-status');
  var pip = document.getElementById('drop-pip');
  var pipSprite = document.getElementById('drop-pip-sprite');
  var bubble = document.getElementById('drop-bubble');
  var areaSelect = document.getElementById('area-select');
  var useArea = document.getElementById('use-area');
  var wholePage = document.getElementById('whole-page');
  var preview = null;
  var currentUrl = '';
  var currentTarget = null;
  var drag = null;
  var suppressClick = false;
  var bubbleTimer = 0;
  var launchPending = false;
  var captureEpoch = 0;
  var captureController = null;
  var previewObjectUrl = '';
  var params = new URLSearchParams(location.search);

  function optionValue(name, allowed, fallback) {
    var value = params.get(name);
    return allowed.indexOf(value) > -1 ? value : fallback;
  }

  var quickOptions = {
    category: optionValue('category', ['auto', 'ecommerce', 'automotive', 'real_estate', 'weather', 'jobs', 'news', 'events', 'travel', 'restaurants', 'recipes', 'finance', 'sports', 'research', 'directory', 'generic'], 'auto'),
    output_format: optionValue('output_format', ['json', 'csv', 'jsonl', 'xlsx', 'sqlite', 'bundle'], 'json'),
    image_mode: optionValue('image_mode', ['links', 'download', 'skip'], 'links'),
    render_mode: optionValue('render_mode', ['auto', 'http', 'browser'], 'auto'),
    max_items: Math.max(1, Math.min(2000, parseInt(params.get('max_items'), 10) || 100)),
    max_pages: Math.max(1, Math.min(200, parseInt(params.get('max_pages'), 10) || 25)),
    target_fields: String(params.get('target_fields') || '').slice(0, 500),
    use_ai: params.get('use_ai') !== '0'
  };

  function requestedFields() {
    var seen = {};
    return quickOptions.target_fields.split(/[,\n]+/).map(function (raw) {
      var name = raw.toLowerCase().trim().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 48);
      if (!name || seen[name]) return null;
      seen[name] = true;
      var type = /image|photo|thumbnail|picture/.test(name) ? 'image' :
        (/price|cost|amount|value/.test(name) ? 'money' : (/url|link/.test(name) ? 'url' : 'auto'));
      return { name: name, type: type, hint: '', required: false };
    }).filter(Boolean).slice(0, 16);
  }

  function normalizeUrl(value) {
    var url = String(value || '').trim();
    if (!url) return '';
    if (!/^https?:\/\//i.test(url)) url = 'https://' + url;
    try {
      var parsed = new URL(url);
      if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') return '';
      return parsed.href;
    } catch (_error) {
      return '';
    }
  }

  function detailFrom(response, fallback) {
    return response.json().then(function (body) { return body.detail || fallback; }).catch(function () { return fallback; });
  }

  function setStatus(message, busy) {
    status.textContent = message;
    status.closest('.status-card').classList.toggle('is-busy', Boolean(busy));
  }

  function say(message, duration) {
    clearTimeout(bubbleTimer);
    bubble.textContent = message;
    bubble.classList.add('on');
    bubbleTimer = setTimeout(function () { bubble.classList.remove('on'); }, duration || 1500);
  }

  function setMotion(motion) {
    pipSprite.setAttribute('data-motion', motion);
  }

  function clearTarget() {
    currentTarget = null;
    outline.classList.remove('on');
  }

  function highlightTarget(element) {
    if (!preview || !element) { clearTarget(); return; }
    currentTarget = element;
    outline.style.left = (element.x / preview.width * 100) + '%';
    outline.style.top = (element.y / preview.height * 100) + '%';
    outline.style.width = (element.width / preview.width * 100) + '%';
    outline.style.height = (element.height / preview.height * 100) + '%';
    outline.classList.add('on');
    setStatus('Target: ' + element.label + ' · ' + element.role, false);
  }

  function elementAt(clientX, clientY) {
    if (!preview || canvas.hidden) return null;
    var rect = image.getBoundingClientRect();
    if (clientX < rect.left || clientX > rect.right || clientY < rect.top || clientY > rect.bottom) return null;
    var x = (clientX - rect.left) / rect.width * preview.width;
    var y = (clientY - rect.top) / rect.height * preview.height;
    var matches = preview.elements.filter(function (element) {
      return x >= element.x && x <= element.x + element.width && y >= element.y && y <= element.y + element.height;
    });
    matches.sort(function (a, b) { return a.width * a.height - b.width * b.height; });
    return matches[0] || null;
  }

  function pipCenter() {
    var rect = pip.getBoundingClientRect();
    return { x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
  }

  function positionPip(left, top) {
    var stageRect = stage.getBoundingClientRect();
    var size = pip.getBoundingClientRect();
    var gutter = 10;
    var x = Math.max(gutter, Math.min(left, stageRect.width - size.width - gutter));
    var y = Math.max(gutter, Math.min(top, stageRect.height - size.height - gutter));
    pip.style.left = x + 'px';
    pip.style.top = y + 'px';
    pip.style.right = 'auto';
    bubble.style.left = x > stageRect.width / 2 ? Math.max(10, x - 205) + 'px' : Math.min(stageRect.width - 190, x + size.width) + 'px';
    bubble.style.right = 'auto';
    bubble.style.top = Math.max(10, Math.min(y + 20, stageRect.height - 70)) + 'px';
  }

  function renderAreaOptions() {
    areaSelect.innerHTML = '';
    var intro = document.createElement('option');
    intro.value = '';
    intro.textContent = preview.elements.length ? 'Choose one of ' + preview.elements.length + ' detected areas' : 'No repeated areas detected';
    areaSelect.appendChild(intro);
    preview.elements.forEach(function (element, index) {
      var option = document.createElement('option');
      option.value = element.element_id;
      option.textContent = (index + 1) + '. ' + element.label + ' · ' + element.role;
      areaSelect.appendChild(option);
    });
    areaSelect.disabled = !preview.elements.length;
    useArea.disabled = true;
  }

  async function loadPreview(rawUrl) {
    var url = normalizeUrl(rawUrl);
    if (!url) { input.focus(); setStatus('Enter a valid http:// or https:// URL.', false); return; }
    var epoch = ++captureEpoch;
    if (captureController) captureController.abort();
    captureController = new AbortController();
    currentUrl = url;
    input.value = url;
    params.set('url', url);
    history.replaceState(null, '', '/overlay?' + params.toString());
    preview = null;
    clearTarget();
    canvas.hidden = true;
    captureState.hidden = false;
    captureState.querySelector('strong').textContent = 'Painting a safe snapshot…';
    captureState.querySelector('span:last-child').textContent = 'Target scripts stay off. Weaver only receives a picture and opaque drop zones.';
    wholePage.disabled = true;
    areaSelect.disabled = true;
    useArea.disabled = true;
    setMotion('weave');
    say('Painting the page safely…', 2200);
    setStatus('Capturing a script-free snapshot. This can take a few seconds.', true);
    var nextObjectUrl = '';
    try {
      var response = await fetch('/api/previews', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: url }),
        signal: captureController.signal
      });
      if (!response.ok) throw new Error(await detailFrom(response, 'Preview request failed with HTTP ' + response.status));
      var nextPreview = await response.json();
      var imageResponse = await fetch(nextPreview.image_url, { signal: captureController.signal, cache: 'no-store' });
      if (!imageResponse.ok) throw new Error('The preview image could not be loaded.');
      var imageBlob = await imageResponse.blob();
      if (epoch !== captureEpoch) return;
      nextObjectUrl = URL.createObjectURL(imageBlob);
      await new Promise(function (resolve, reject) {
        image.onload = resolve;
        image.onerror = function () { reject(new Error('The preview image could not be loaded.')); };
        image.src = nextObjectUrl;
      });
      if (epoch !== captureEpoch) { URL.revokeObjectURL(nextObjectUrl); return; }
      if (previewObjectUrl) URL.revokeObjectURL(previewObjectUrl);
      previewObjectUrl = nextObjectUrl;
      preview = nextPreview;
      canvas.hidden = false;
      captureState.hidden = true;
      wholePage.disabled = false;
      renderAreaOptions();
      setMotion('idle');
      say(preview.elements.length ? 'Drop me on a card!' : 'I can still scrape the whole page.', 2200);
      setStatus(preview.title + ' · ' + preview.elements.length + ' drop zones found', false);
    } catch (error) {
      if (nextObjectUrl && nextObjectUrl !== previewObjectUrl) URL.revokeObjectURL(nextObjectUrl);
      if (error && error.name === 'AbortError') return;
      if (epoch !== captureEpoch) return;
      captureState.hidden = false;
      captureState.querySelector('strong').textContent = 'Snapshot unavailable';
      captureState.querySelector('span:last-child').textContent = error.message;
      setMotion('idle');
      say('That page stayed out of reach.', 2200);
      setStatus(error.message, false);
    } finally {
      if (epoch === captureEpoch) captureController = null;
    }
  }

  async function startRun(element) {
    if (launchPending || !currentUrl || !preview) return;
    launchPending = true;
    setMotion('happy');
    say(element ? 'Got it—back to the loom!' : 'I’ll map the whole page!', 2200);
    setStatus(element ? 'Weaver grabbed ' + element.label + '. Starting the scraper…' : 'Starting a whole-page scraper…', true);
    var payload = {
      urls: [currentUrl],
      options: {
        category: quickOptions.category,
        output_format: quickOptions.output_format,
        image_mode: quickOptions.image_mode,
        render_mode: quickOptions.render_mode,
        max_items: quickOptions.max_items,
        max_pages: quickOptions.max_pages,
        use_ai: quickOptions.use_ai,
        target_intent: '',
        requested_fields: requestedFields()
      }
    };
    if (element) payload.selection = { preview_id: preview.preview_id, element_id: element.element_id };
    try {
      var response = await fetch('/api/runs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      });
      if (!response.ok) throw new Error(await detailFrom(response, 'Run request failed with HTTP ' + response.status));
      var created = await response.json();
      setTimeout(function () { window.location.assign('/?run=' + encodeURIComponent(created.id) + '&stay=1&from=overlay'); }, 520);
    } catch (error) {
      launchPending = false;
      setMotion('idle');
      say('The thread slipped. Try again?', 2200);
      setStatus(error.message, false);
    }
  }

  pip.addEventListener('pointerdown', function (event) {
    if (!event.isPrimary || (event.pointerType === 'mouse' && event.button !== 0)) return;
    var pipRect = pip.getBoundingClientRect();
    var stageRect = stage.getBoundingClientRect();
    drag = {
      id: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originLeft: pipRect.left - stageRect.left,
      originTop: pipRect.top - stageRect.top,
      offsetX: event.clientX - pipRect.left,
      offsetY: event.clientY - pipRect.top,
      moved: false,
      target: null
    };
    pip.setPointerCapture(event.pointerId);
  });

  pip.addEventListener('pointermove', function (event) {
    if (!drag || drag.id !== event.pointerId) return;
    var distance = Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY);
    var threshold = event.pointerType === 'touch' ? 10 : 6;
    if (!drag.moved && distance < threshold) return;
    if (!drag.moved) {
      drag.moved = true;
      pip.classList.add('is-dragging');
      setMotion('crawl');
    }
    event.preventDefault();
    var stageRect = stage.getBoundingClientRect();
    positionPip(event.clientX - stageRect.left - drag.offsetX, event.clientY - stageRect.top - drag.offsetY);
    var center = pipCenter();
    drag.target = elementAt(center.x, center.y);
    if (drag.target) highlightTarget(drag.target);
    else clearTarget();
    var scrollRect = scroll.getBoundingClientRect();
    if (event.clientY < scrollRect.top + 42) scroll.scrollBy(0, -18);
    else if (event.clientY > scrollRect.bottom - 42) scroll.scrollBy(0, 18);
  });

  function finishDrag(event, cancelled) {
    if (!drag || drag.id !== event.pointerId) return;
    var completed = drag;
    drag = null;
    pip.classList.remove('is-dragging');
    if (pip.hasPointerCapture(event.pointerId)) pip.releasePointerCapture(event.pointerId);
    if (!completed.moved) return;
    suppressClick = true;
    setTimeout(function () { suppressClick = false; }, 0);
    if (cancelled) {
      positionPip(completed.originLeft, completed.originTop);
      clearTarget();
      setMotion('idle');
      say('Back where I started.', 1200);
      return;
    }
    if (completed.target) startRun(completed.target);
    else {
      setMotion('idle');
      say('Try a repeated card or row.', 1800);
      setStatus('No drop zone there. Look for the green outline.', false);
    }
  }

  pip.addEventListener('pointerup', function (event) { finishDrag(event, false); });
  pip.addEventListener('pointercancel', function (event) { finishDrag(event, true); });
  pip.addEventListener('lostpointercapture', function (event) { if (drag) finishDrag(event, true); });
  pip.addEventListener('click', function (event) {
    if (suppressClick) { event.preventDefault(); event.stopImmediatePropagation(); return; }
    setMotion('happy');
    say('Tiny high-five!', 1100);
    setTimeout(function () { if (!launchPending) setMotion('idle'); }, 850);
  }, true);

  pip.addEventListener('keydown', function (event) {
    var movement = event.shiftKey ? 32 : 12;
    var dx = 0, dy = 0;
    if (event.key === 'ArrowLeft') dx = -movement;
    else if (event.key === 'ArrowRight') dx = movement;
    else if (event.key === 'ArrowUp') dy = -movement;
    else if (event.key === 'ArrowDown') dy = movement;
    else if (event.key.toLowerCase() === 's') {
      event.preventDefault();
      var center = pipCenter();
      var target = elementAt(center.x, center.y);
      if (target) { highlightTarget(target); startRun(target); }
      else say('Move me over a green area first.', 1700);
      return;
    } else if (event.key === 'Escape') {
      clearTarget(); setMotion('idle'); return;
    } else return;
    event.preventDefault();
    var pipRect = pip.getBoundingClientRect();
    var stageRect = stage.getBoundingClientRect();
    positionPip(pipRect.left - stageRect.left + dx, pipRect.top - stageRect.top + dy);
    var point = pipCenter();
    var candidate = elementAt(point.x, point.y);
    if (candidate) highlightTarget(candidate); else clearTarget();
    setMotion('crawl');
    setTimeout(function () { if (!launchPending) setMotion('idle'); }, 360);
  });

  areaSelect.addEventListener('change', function () {
    var element = preview && preview.elements.find(function (item) { return item.element_id === areaSelect.value; });
    useArea.disabled = !element;
    if (element) {
      highlightTarget(element);
      var imageRect = image.getBoundingClientRect();
      scroll.scrollTo({ top: Math.max(0, element.y / preview.height * imageRect.height - 80), behavior: 'smooth' });
    } else clearTarget();
  });
  useArea.addEventListener('click', function () {
    var element = preview && preview.elements.find(function (item) { return item.element_id === areaSelect.value; });
    if (element) startRun(element);
  });
  wholePage.addEventListener('click', function () { startRun(null); });
  form.addEventListener('submit', function (event) { event.preventDefault(); loadPreview(input.value); });
  document.querySelectorAll('[data-sample]').forEach(function (button) {
    button.addEventListener('click', function () { input.value = button.getAttribute('data-sample'); loadPreview(input.value); });
  });
  pip.addEventListener('pointerenter', function () { if (!drag) say('Drag me onto a repeated card.', 1500); });

  var initial = params.get('url') || 'https://books.toscrape.com/';
  input.value = initial;
  loadPreview(initial);
})();
