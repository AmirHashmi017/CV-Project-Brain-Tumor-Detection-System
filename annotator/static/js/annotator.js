let images = [];
let currentImage = null;
let annotations = [];
let selectedClass = 0;
let mode = 'bbox'; // 'bbox' or 'polygon'
let points = [];
let isDrawing = false;
let zoom = 1.0;
let classNames = ["glioma", "meningioma", "no_tumor", "pituitary"];

const CLS_COLORS = ["#dc3232","#32c850","#3264e6","#e6c828","#b432dc","#32dce6","#e67832","#e63282"];

// Elements
const imgEl = document.getElementById('base-img');
const cv = document.getElementById('main-canvas');
const ctx = cv.getContext('2d');
const imgListEl = document.getElementById('image-list');
const annCountEl = document.getElementById('ann-count');
const statusBadge = document.getElementById('status-badge');
const filenameEl = document.getElementById('current-filename');

// Initialize
async function init() {
    await fetchImages();
    setupEventListeners();
    if (images.length > 0) loadItem(images[0]);
}

async function fetchImages() {
    const res = await fetch('/api/images');
    images = await res.json();
    renderImageList();
}

function renderImageList() {
    imgListEl.innerHTML = images.map(img => `
        <div class="image-item ${img === currentImage ? 'active' : ''}" onclick="loadItem('${img}')">
            ${img}
        </div>
    `).join('');
}

async function loadItem(filename) {
    currentImage = filename;
    renderImageList();
    filenameEl.textContent = filename;
    statusBadge.textContent = "Loading...";
    
    const res = await fetch(`/api/load_annotations/${filename}`);
    const data = await res.json();
    
    imgEl.src = `/images/${filename}`;
    imgEl.onload = () => {
        cv.width = imgEl.naturalWidth;
        cv.height = imgEl.naturalHeight;
        annotations = data.annotations || [];
        statusBadge.textContent = "Ready";
        draw();
    };
}

function setupEventListeners() {
    // Canvas events
    cv.addEventListener('mousedown', startDrawing);
    cv.addEventListener('mousemove', moveDrawing);
    cv.addEventListener('mouseup', endDrawing);
    cv.addEventListener('contextmenu', e => {
        e.preventDefault();
        if (mode === 'polygon' && points.length > 2) finishPolygon();
    });

    // Toolbar events
    document.getElementById('mode-bbox').onclick = () => setMode('bbox');
    document.getElementById('mode-poly').onclick = () => setMode('polygon');
    
    // Upload logic
    const fileInput = document.getElementById('file-input');
    const btnUpload = document.getElementById('btn-upload');
    
    btnUpload.onclick = () => fileInput.click();
    fileInput.onchange = async () => {
        if (fileInput.files.length === 0) return;
        statusBadge.textContent = "Uploading...";
        const formData = new FormData();
        for (let file of fileInput.files) formData.append('files', file);
        try {
            const res = await fetch('/api/upload', { method: 'POST', body: formData });
            const result = await res.json();
            if (result.success) {
                await fetchImages();
                statusBadge.textContent = "Uploaded ✓";
                if (result.uploaded.length > 0) loadItem(result.uploaded[0]);
            }
        } catch (err) { statusBadge.textContent = "Upload Failed"; }
        fileInput.value = '';
    };

    // Custom Class
    const classInput = document.getElementById('custom-class-input');
    const btnAddClass = document.getElementById('btn-add-class');
    btnAddClass.onclick = () => {
        const name = classInput.value.trim();
        if (name && !classNames.includes(name)) {
            classNames.push(name);
            renderClasses();
            classInput.value = '';
        }
    };

    // Sidebar actions
    document.getElementById('btn-save').onclick = saveAnnotations;
    document.getElementById('btn-undo').onclick = undo;
    document.getElementById('btn-clear').onclick = clearAll;

    // Downloads
    document.getElementById('dl-img').onclick = downloadImage;
    document.getElementById('dl-txt').onclick = downloadYOLO;
    document.getElementById('dl-json').onclick = downloadJSON;

    // Class selection
    setupClassSelection();

    // Hotkeys
    window.addEventListener('keydown', e => {
        if (e.target.tagName === 'INPUT') return;
        if (e.key.toLowerCase() === 'b') setMode('bbox');
        if (e.key.toLowerCase() === 'p') setMode('polygon');
        if (e.key.toLowerCase() === 's') saveAnnotations();
        if (e.ctrlKey && e.key === 'z') undo();
        if (e.key >= '1' && e.key <= '9') {
            const idx = parseInt(e.key) - 1;
            const item = document.querySelector(`.class-item[data-id="${idx}"]`);
            if (item) item.click();
        }
    });
}

function setupClassSelection() {
    document.querySelectorAll('.class-item').forEach(item => {
        item.onclick = () => {
            document.querySelectorAll('.class-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            selectedClass = parseInt(item.dataset.id);
        };
    });
}

function renderClasses() {
    const list = document.getElementById('class-list');
    list.innerHTML = classNames.map((c, i) => `
        <div class="class-item ${i === selectedClass ? 'active' : ''}" data-id="${i}">
            <span class="color-dot" style="background-color: ${CLS_COLORS[i % CLS_COLORS.length]}"></span>
            <span class="name">${c}</span>
            <span class="key">${i + 1}</span>
        </div>
    `).join('');
    setupClassSelection();
}

function setMode(newMode) {
    mode = newMode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(`mode-${mode === 'bbox' ? 'bbox' : 'poly'}`).classList.add('active');
    points = [];
    draw();
}

function startDrawing(e) {
    if (e.button !== 0) return;
    const pos = getMousePos(e);
    if (mode === 'bbox') {
        isDrawing = true;
        points = [pos, pos];
    } else {
        points.push(pos);
    }
    draw();
}

function moveDrawing(e) {
    const pos = getMousePos(e);
    if (mode === 'bbox' && isDrawing) {
        points[1] = pos;
        draw();
    } else if (mode === 'polygon' && points.length > 0) {
        draw();
        ctx.strokeStyle = CLS_COLORS[selectedClass % CLS_COLORS.length];
        ctx.setLineDash([5, 5]);
        ctx.beginPath(); ctx.moveTo(points[points.length-1][0], points[points.length-1][1]);
        ctx.lineTo(pos[0], pos[1]); ctx.stroke(); ctx.setLineDash([]);
    }
}

function endDrawing(e) {
    if (mode === 'bbox' && isDrawing) {
        isDrawing = false;
        const pos = getMousePos(e);
        points[1] = pos;
        if (Math.abs(points[0][0] - points[1][0]) > 5) {
            annotations.push({ type: 'bbox', class_id: selectedClass, points: [...points] });
        }
        points = [];
        draw();
    }
}

function finishPolygon() {
    annotations.push({ type: 'polygon', class_id: selectedClass, points: [...points] });
    points = [];
    draw();
}

function undo() {
    if (points.length > 0) points.pop();
    else annotations.pop();
    draw();
}

function clearAll() {
    if (confirm("Clear all annotations for this image?")) {
        annotations = []; points = []; draw();
    }
}

async function saveAnnotations() {
    if (!currentImage) return;
    statusBadge.textContent = "Saving...";
    try {
        const res = await fetch('/api/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ filename: currentImage, annotations })
        });
        const result = await res.json();
        if (result.success) {
            statusBadge.textContent = "Saved ✓";
            setTimeout(() => statusBadge.textContent = "Ready", 2000);
        }
    } catch (err) { statusBadge.textContent = "Save Failed"; }
}

function downloadImage() {
    const tempCv = document.createElement('canvas');
    tempCv.width = cv.width; tempCv.height = cv.height;
    const tempCtx = tempCv.getContext('2d');
    tempCtx.drawImage(imgEl, 0, 0);
    tempCtx.drawImage(cv, 0, 0);
    const link = document.createElement('a');
    link.download = currentImage.split('.')[0] + '_annotated.png';
    link.href = tempCv.toDataURL();
    link.click();
}

function downloadYOLO() {
    let txt = "";
    const w = cv.width, h = cv.height;
    annotations.forEach(ann => {
        if (ann.type === 'bbox') {
            const [[x1,y1],[x2,y2]] = ann.points;
            const xmin=Math.min(x1,x2), xmax=Math.max(x1,x2), ymin=Math.min(y1,y2), ymax=Math.max(y1,y2);
            const cx = ((xmin+xmax)/2)/w, cy = ((ymin+ymax)/2)/h, bw = (xmax-xmin)/w, bh = (ymax-ymin)/h;
            txt += `${ann.class_id} ${cx.toFixed(6)} ${cy.toFixed(6)} ${bw.toFixed(6)} ${bh.toFixed(6)}\n`;
        } else {
            const pts = ann.points.map(p => `${(p[0]/w).toFixed(6)} ${(p[1]/h).toFixed(6)}`).join(' ');
            txt += `${ann.class_id} ${pts}\n`;
        }
    });
    const blob = new Blob([txt], {type: 'text/plain'});
    const link = document.createElement('a');
    link.download = currentImage.split('.')[0] + '.txt';
    link.href = URL.createObjectURL(blob);
    link.click();
}

function downloadJSON() {
    const data = { image: currentImage, width: cv.width, height: cv.height, classes: classNames, annotations };
    const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
    const link = document.createElement('a');
    link.download = currentImage.split('.')[0] + '.json';
    link.href = URL.createObjectURL(blob);
    link.click();
}

function getMousePos(e) {
    const rect = cv.getBoundingClientRect();
    const scaleX = cv.width / rect.width;
    const scaleY = cv.height / rect.height;
    return [Math.round((e.clientX - rect.left) * scaleX), Math.round((e.clientY - rect.top) * scaleY)];
}

function draw() {
    ctx.clearRect(0, 0, cv.width, cv.height);
    
    annotations.forEach(ann => {
        const color = CLS_COLORS[ann.class_id % CLS_COLORS.length];
        const lbl = classNames[ann.class_id] || ann.class_id;
        ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.fillStyle = color + '22';
        
        let tx, ty;
        if (ann.type === 'bbox') {
            const [p1, p2] = ann.points;
            const x = Math.min(p1[0], p2[0]), y = Math.min(p1[1], p2[1]), w = Math.abs(p2[0]-p1[0]), h = Math.abs(p2[1]-p1[1]);
            ctx.strokeRect(x,y,w,h); ctx.fillRect(x,y,w,h);
            tx = x; ty = y;
        } else {
            ctx.beginPath(); ctx.moveTo(ann.points[0][0], ann.points[0][1]);
            ann.points.forEach(p => ctx.lineTo(p[0], p[1])); ctx.closePath();
            ctx.stroke(); ctx.fill();
            tx = ann.points[0][0]; ty = ann.points[0][1];
        }
        drawLabel(lbl, tx, ty, color);
    });

    if (points.length > 0) {
        const color = CLS_COLORS[selectedClass % CLS_COLORS.length];
        ctx.strokeStyle = color; ctx.lineWidth = 2;
        if (mode === 'bbox') {
            const [p1, p2] = points;
            ctx.strokeRect(p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1]);
        } else {
            ctx.beginPath(); ctx.moveTo(points[0][0], points[0][1]);
            points.forEach(p => ctx.lineTo(p[0], p[1])); ctx.stroke();
            points.forEach(p => {
                ctx.beginPath(); ctx.arc(p[0], p[1], 4, 0, Math.PI*2);
                ctx.fillStyle = color; ctx.fill();
            });
        }
    }
    annCountEl.textContent = `${annotations.length} annotations`;
}

function drawLabel(txt, x, y, color) {
    ctx.font = 'bold 12px monospace';
    const tw = ctx.measureText(txt).width;
    const ty = Math.max(y - 5, 15);
    ctx.fillStyle = color;
    ctx.fillRect(x, ty - 13, tw + 8, 16);
    ctx.fillStyle = '#fff';
    ctx.fillText(txt, x + 4, ty);
}

init();
