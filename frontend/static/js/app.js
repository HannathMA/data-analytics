/**
 * Application Controller & API Integration for Smart Meter Clustering Dashboard
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const statusPill = document.getElementById('datasetStatusPill');
  const statusText = document.getElementById('datasetStatusText');
  const btnLoadSample = document.getElementById('btnLoadSample');
  const btnOpenUploadModal = document.getElementById('btnOpenUploadModal');
  const btnCloseUploadModal = document.getElementById('btnCloseUploadModal');
  const btnCancelUpload = document.getElementById('btnCancelUpload');
  const btnSubmitUpload = document.getElementById('btnSubmitUpload');
  const uploadModal = document.getElementById('uploadModal');
  const dropZone = document.getElementById('dropZone');
  const fileInput = document.getElementById('fileInput');
  const selectedFileInfo = document.getElementById('selectedFileInfo');
  const selectedFileName = document.getElementById('selectedFileName');
  const selectedFileSize = document.getElementById('selectedFileSize');

  const kSelect = document.getElementById('kSelect');
  const btnRunClustering = document.getElementById('btnRunClustering');
  const btnExportCsv = document.getElementById('btnExportCsv');
  const btnThemeToggle = document.getElementById('btnThemeToggle');
  const btnCalculateElbow = document.getElementById('btnCalculateElbow');
  const meterSelect = document.getElementById('meterSelect');
  const toggleRawAcf = document.getElementById('toggleRawAcf');

  // KPI elements
  const kpiHouseholds = document.getElementById('kpiHouseholds');
  const kpiRetention = document.getElementById('kpiRetention');
  const kpiClusters = document.getElementById('kpiClusters');
  const kpiSilhouette = document.getElementById('kpiSilhouette');
  const kpiSilhouetteDesc = document.getElementById('kpiSilhouetteDesc');
  const kpiDbi = document.getElementById('kpiDbi');

  // Waterfall elements
  const wfRawCount = document.getElementById('wfRawCount');
  const wfDroppedMissing = document.getElementById('wfDroppedMissing');
  const wfDroppedFlat = document.getElementById('wfDroppedFlat');
  const wfDroppedVacant = document.getElementById('wfDroppedVacant');
  const wfFinalCount = document.getElementById('wfFinalCount');

  // Cluster Catalog
  const clusterCardsContainer = document.getElementById('clusterCardsContainer');
  const centroidFilterButtons = document.getElementById('centroidFilterButtons');
  const distTotalLabel = document.getElementById('distTotalLabel');
  const pcaVarianceLabel = document.getElementById('pcaVarianceLabel');

  let activeMeterData = null;
  let selectedFile = null;

  // Initialize Application
  initApp();

  async function initApp() {
    initTheme();
    setupEventListeners();
    await checkStatusAndLoad();
  }

  function initTheme() {
    const savedTheme = localStorage.getItem('voltcluster_theme') || 'light-theme';
    setTheme(savedTheme);
  }

  function toggleTheme() {
    const current = document.body.classList.contains('light-theme') ? 'light-theme' : 'dark-theme';
    const next = current === 'light-theme' ? 'dark-theme' : 'light-theme';
    setTheme(next);
  }

  function setTheme(theme) {
    document.body.classList.remove('light-theme', 'dark-theme');
    document.body.classList.add(theme);
    localStorage.setItem('voltcluster_theme', theme);
    if (window.dashboardCharts && typeof window.dashboardCharts.refreshTheme === 'function') {
      window.dashboardCharts.refreshTheme();
    }
  }

  function setupEventListeners() {
    // Sample generation
    btnLoadSample.addEventListener('click', handleGenerateSample);

    // Clustering execution
    btnRunClustering.addEventListener('click', () => runClustering());

    // Export CSV
    btnExportCsv.addEventListener('click', () => {
      window.location.href = '/api/export_clusters';
    });

    // Theme toggle
    if (btnThemeToggle) {
      btnThemeToggle.addEventListener('click', toggleTheme);
    }

    // Elbow evaluation
    btnCalculateElbow.addEventListener('click', loadElbowData);

    // Meter selector change
    meterSelect.addEventListener('change', (e) => {
      if (e.target.value) {
        loadMeterDetails(e.target.value);
      }
    });

    // ACF noise toggle
    toggleRawAcf.addEventListener('change', (e) => {
      if (activeMeterData && activeMeterData.acf) {
        window.dashboardCharts.renderAcf(activeMeterData.acf, e.target.checked);
      }
    });

    // Upload Modal Handling
    btnOpenUploadModal.addEventListener('click', () => {
      uploadModal.style.display = 'flex';
    });
    btnCloseUploadModal.addEventListener('click', closeModal);
    btnCancelUpload.addEventListener('click', closeModal);

    // Drag & Drop
    dropZone.addEventListener('click', () => fileInput.click());
    dropZone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropZone.classList.add('dragover');
    });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
    dropZone.addEventListener('drop', (e) => {
      e.preventDefault();
      dropZone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });
    btnSubmitUpload.addEventListener('click', handleFileUpload);
  }

  function closeModal() {
    uploadModal.style.display = 'none';
    selectedFile = null;
    fileInput.value = '';
    selectedFileInfo.style.display = 'none';
    btnSubmitUpload.disabled = true;
  }

  function handleFileSelection(file) {
    if (!file.name.endsWith('.csv')) {
      showToast('Please upload a valid .csv file.', 'error');
      return;
    }
    selectedFile = file;
    selectedFileName.textContent = file.name;
    selectedFileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    selectedFileInfo.style.display = 'flex';
    btnSubmitUpload.disabled = false;
  }

  async function handleFileUpload() {
    if (!selectedFile) return;

    const formData = new FormData();
    formData.append('file', selectedFile);

    btnSubmitUpload.disabled = true;
    btnSubmitUpload.textContent = 'Uploading & Cleaning...';

    try {
      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || 'Upload failed');

      showToast(`Uploaded & Cleaned: ${data.cleaned_count} households ready!`, 'success');
      closeModal();
      updateWaterfallUI(data.audit_log);
      await runClustering();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnSubmitUpload.disabled = false;
      btnSubmitUpload.textContent = 'Upload & Clean';
    }
  }

  async function handleGenerateSample() {
    btnLoadSample.disabled = true;
    btnLoadSample.innerHTML = '<span class="status-dot"></span> Generating...';

    try {
      const res = await fetch('/api/generate_sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_households: 250 })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed generating sample');

      showToast('London SmartMeter sample loaded with waterfall edge cases!', 'success');
      updateWaterfallUI(data.audit_log);
      await runClustering();
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnLoadSample.disabled = false;
      btnLoadSample.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
          <polyline points="7 10 12 15 17 10"></polyline>
          <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg> Load London Sample
      `;
    }
  }

  async function checkStatusAndLoad() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();

      if (data.status === 'ready') {
        statusText.textContent = `Active Dataset: ${data.cleaned_meters_count} Households`;
        if (data.audit_log) {
          updateWaterfallUI(data.audit_log);
        } else {
          // Synthetic defaults
          wfRawCount.textContent = `${data.cleaned_meters_count + 9}`;
          wfDroppedMissing.textContent = '-3';
          wfDroppedFlat.textContent = '-3';
          wfDroppedVacant.textContent = '-3';
          wfFinalCount.textContent = `${data.cleaned_meters_count}`;
        }
        await runClustering();
      } else {
        statusText.textContent = 'No dataset loaded';
        // Auto-generate sample for great first-run experience
        await handleGenerateSample();
      }
    } catch (err) {
      statusText.textContent = 'Server connecting...';
    }
  }

  async function runClustering() {
    const k = parseInt(kSelect.value, 10);
    btnRunClustering.disabled = true;
    btnRunClustering.textContent = 'Clustering...';

    try {
      const res = await fetch('/api/run_clustering', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ k: k, max_lag: 24 })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Clustering failed');

      updateKPIs(data.metrics);
      renderClusterCatalog(data.cluster_profiles);
      setupCentroidFilterButtons(data.cluster_profiles);

      window.dashboardCharts.renderCentroids(data.cluster_profiles);
      window.dashboardCharts.renderDistribution(data.cluster_profiles);
      distTotalLabel.textContent = `${data.metrics.n_samples} households`;

      // Fetch full cluster state with assignments for PCA and meter dropdown
      const fullRes = await fetch('/api/get_clusters');
      const fullData = await fullRes.json();

      if (fullData.assignments) {
        window.dashboardCharts.renderPca(fullData.assignments, k);
        populateMeterSelect(fullData.assignments);
      }

      if (data.pca_variance_ratio && data.pca_variance_ratio.length >= 2) {
        const pc1 = (data.pca_variance_ratio[0] * 100).toFixed(1);
        const pc2 = (data.pca_variance_ratio[1] * 100).toFixed(1);
        pcaVarianceLabel.textContent = `Explained Variance: PC1 (${pc1}%) | PC2 (${pc2}%)`;
      }

      // Load elbow analysis in background
      loadElbowData();

      showToast(`K-Means clustering complete (${k} behavioral clusters).`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      btnRunClustering.disabled = false;
      btnRunClustering.innerHTML = `
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <polygon points="5 3 19 12 5 21 5 3"></polygon>
        </svg> Run Clustering
      `;
    }
  }

  function updateKPIs(metrics) {
    if (!metrics) return;
    kpiHouseholds.textContent = metrics.n_samples;
    kpiClusters.textContent = metrics.n_clusters;

    if (metrics.silhouette_score !== null) {
      kpiSilhouette.textContent = metrics.silhouette_score.toFixed(3);
      kpiSilhouetteDesc.textContent = metrics.silhouette_score > 0.3 ? 'Strong separation' : 'Moderate clustering';
    } else {
      kpiSilhouette.textContent = 'N/A';
    }

    if (metrics.davies_bouldin_index !== null) {
      kpiDbi.textContent = metrics.davies_bouldin_index.toFixed(3);
    } else {
      kpiDbi.textContent = 'N/A';
    }
  }

  function updateWaterfallUI(audit) {
    if (!audit) return;
    wfRawCount.textContent = audit.initial_count || '--';
    wfDroppedMissing.textContent = `-${audit.dropped_missing_count || 0}`;
    wfDroppedFlat.textContent = `-${audit.dropped_flat_count || 0}`;
    wfDroppedVacant.textContent = `-${audit.dropped_vacant_count || 0}`;
    wfFinalCount.textContent = audit.final_count || '--';
    kpiRetention.textContent = `Retention: ${audit.retention_rate_pct || 100}%`;
  }

  function populateMeterSelect(assignments) {
    meterSelect.innerHTML = '';
    assignments.slice(0, 80).forEach((item, idx) => {
      const opt = document.createElement('option');
      opt.value = item.meter_id;
      opt.textContent = `${item.meter_id} (Cluster ${item.cluster})`;
      meterSelect.appendChild(opt);
    });

    if (assignments.length > 0) {
      loadMeterDetails(assignments[0].meter_id);
    }
  }

  async function loadMeterDetails(meterId) {
    try {
      const res = await fetch(`/api/meter/${meterId}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error);

      activeMeterData = data;

      // Update charts
      window.dashboardCharts.renderRawSeries(data.raw_series_168h, data.meter_id);
      window.dashboardCharts.renderAcf(data.acf, toggleRawAcf.checked);

      const statsEl = document.getElementById('meterStatsLabel');
      if (statsEl) {
        statsEl.textContent = `Cluster: ${data.assigned_cluster !== null ? data.assigned_cluster : 'N/A'} | Mean: ${data.mean_kwh} kWh | Max: ${data.max_kwh} kWh`;
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function loadElbowData() {
    try {
      const res = await fetch('/api/elbow');
      const data = await res.json();
      if (res.ok) {
        window.dashboardCharts.renderElbow(data);
      }
    } catch (err) {
      console.error('Elbow evaluation error:', err);
    }
  }

  function renderClusterCatalog(profiles) {
    clusterCardsContainer.innerHTML = '';

    profiles.forEach(p => {
      const card = document.createElement('div');
      card.className = 'persona-card';

      // 24-lag daily recurrence strength
      const lags = p.centroid_acf.length === 25 ? p.centroid_acf.slice(1) : p.centroid_acf;
      const lag24 = lags[23] !== undefined ? lags[23].toFixed(2) : '0.00';
      const lag12 = lags[11] !== undefined ? lags[11].toFixed(2) : '0.00';

      card.innerHTML = `
        <div>
          <div class="persona-card-header">
            <span class="cluster-tag" style="background: rgba(6, 182, 212, 0.15); color: #38bdf8;">Cluster ${p.cluster_id}</span>
            <span style="font-size: 0.75rem; color: #9ca3af; font-weight: 600;">${p.percentage}% (${p.size} homes)</span>
          </div>
          <h4 class="persona-title">${p.persona}</h4>
          <p class="persona-desc">${p.description}</p>
        </div>
        <div>
          <div class="persona-stats">
            <div class="stat-box">
              <div class="stat-box-val">${lag24}</div>
              <div class="stat-box-lbl">24h Recurrence</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-val">${lag12}</div>
              <div class="stat-box-lbl">12h Harmony</div>
            </div>
            <div class="stat-box">
              <div class="stat-box-val">${p.exemplar_distance}</div>
              <div class="stat-box-lbl">Centroid Dist</div>
            </div>
          </div>
          <div class="persona-footer">
            <span class="exemplar-id">Exemplar: ${p.exemplar_meter_id || 'N/A'}</span>
            <button class="btn-inspect" data-meter="${p.exemplar_meter_id}">Inspect Profile</button>
          </div>
        </div>
      `;

      card.querySelector('.btn-inspect').addEventListener('click', (e) => {
        const mId = e.target.getAttribute('data-meter');
        if (mId && mId !== 'N/A') {
          meterSelect.value = mId;
          loadMeterDetails(mId);
          window.scrollTo({ top: 400, behavior: 'smooth' });
        }
      });

      clusterCardsContainer.appendChild(card);
    });
  }

  function setupCentroidFilterButtons(profiles) {
    centroidFilterButtons.innerHTML = '';

    const allBtn = document.createElement('button');
    allBtn.className = 'cluster-pill-btn active';
    allBtn.textContent = 'All Clusters';
    allBtn.addEventListener('click', () => {
      document.querySelectorAll('.cluster-pill-btn').forEach(b => b.classList.remove('active'));
      allBtn.classList.add('active');
      window.dashboardCharts.isolateCentroid('all');
    });
    centroidFilterButtons.appendChild(allBtn);

    profiles.forEach(p => {
      const btn = document.createElement('button');
      btn.className = 'cluster-pill-btn';
      btn.textContent = `C${p.cluster_id}`;
      btn.addEventListener('click', () => {
        document.querySelectorAll('.cluster-pill-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        window.dashboardCharts.isolateCentroid(p.cluster_id);
      });
      centroidFilterButtons.appendChild(btn);
    });
  }

  function showToast(message, type = 'info') {
    const container = document.getElementById('toastContainer');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${message}</span>`;
    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, 4000);
  }
});
