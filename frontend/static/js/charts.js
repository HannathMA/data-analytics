/**
 * Chart.js Visualizations Manager for Smart Meter ACF Clustering
 * Supports responsive Light and Dark themes
 */

const CLUSTER_COLORS = [
  '#0284c7', '#7c3aed', '#059669', '#d97706', '#e11d48',
  '#2563eb', '#db2777', '#0d9488', '#ca8a04', '#9333ea',
  '#4f46e5', '#475569', '#65a30d', '#c026d3', '#0891b2'
];

class DashboardCharts {
  constructor() {
    this.rawChart = null;
    this.acfChart = null;
    this.centroidsChart = null;
    this.distributionChart = null;
    this.elbowChart = null;
    this.pcaChart = null;

    // Cache last datasets for fast live theme refresh
    this.lastRawArgs = null;
    this.lastAcfArgs = null;
    this.lastCentroidsArgs = null;
    this.lastDistributionArgs = null;
    this.lastElbowArgs = null;
    this.lastPcaArgs = null;
  }

  getThemeColors() {
    const isLight = document.body.classList.contains('light-theme');
    return {
      isLight,
      gridColor: isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.05)',
      tickColor: isLight ? '#64748b' : '#9ca3af',
      legendColor: isLight ? '#334155' : '#9ca3af',
      tooltipBg: isLight ? '#ffffff' : '#111827',
      tooltipTitle: isLight ? '#0f172a' : '#f3f4f6',
      tooltipBody: isLight ? '#0284c7' : '#06b6d4',
      tooltipBorder: isLight ? '#cbd5e1' : 'rgba(255,255,255,0.1)',
      doughnutBorder: isLight ? '#ffffff' : '#111827',
      rawGradientStart: isLight ? 'rgba(2, 132, 199, 0.25)' : 'rgba(6, 182, 212, 0.4)',
      rawGradientEnd: isLight ? 'rgba(2, 132, 199, 0.0)' : 'rgba(6, 182, 212, 0.0)',
      rawLineColor: isLight ? '#0284c7' : '#06b6d4',
      noiseBarColor: isLight ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)',
    };
  }

  refreshTheme() {
    if (this.lastRawArgs) this.renderRawSeries(...this.lastRawArgs);
    if (this.lastAcfArgs) this.renderAcf(...this.lastAcfArgs);
    if (this.lastCentroidsArgs) this.renderCentroids(...this.lastCentroidsArgs);
    if (this.lastDistributionArgs) this.renderDistribution(...this.lastDistributionArgs);
    if (this.lastElbowArgs) this.renderElbow(...this.lastElbowArgs);
    if (this.lastPcaArgs) this.renderPca(...this.lastPcaArgs);
  }

  /**
   * 1. 168-Hour Raw Household Electricity Series Chart
   */
  renderRawSeries(series, meterId = 'Sample Household') {
    this.lastRawArgs = [series, meterId];
    const ctx = document.getElementById('rawSeriesChart');
    if (!ctx) return;

    const theme = this.getThemeColors();
    const labels = Array.from({ length: series.length }, (_, i) => {
      const dayIdx = Math.floor(i / 24);
      const dayNames = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
      const hour = i % 24;
      return `${dayNames[dayIdx] || 'D'} ${hour.toString().padStart(2, '0')}:00`;
    });

    if (this.rawChart) {
      this.rawChart.destroy();
    }

    const gradient = ctx.getContext('2d').createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, theme.rawGradientStart);
    gradient.addColorStop(1, theme.rawGradientEnd);

    this.rawChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: `${meterId} Consumption (kWh)`,
          data: series,
          borderColor: theme.rawLineColor,
          borderWidth: 2,
          backgroundColor: gradient,
          fill: true,
          tension: 0.25,
          pointRadius: 0,
          pointHoverRadius: 5,
          pointHoverBackgroundColor: '#fff',
          pointHoverBorderColor: theme.rawLineColor
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            labels: { color: theme.legendColor, font: { family: 'Plus Jakarta Sans', size: 12 } }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            bodyColor: theme.tooltipBody,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            padding: 10
          }
        },
        scales: {
          x: {
            grid: { color: theme.gridColor },
            ticks: {
              color: theme.tickColor,
              maxTicksLimit: 7,
              font: { family: 'JetBrains Mono', size: 11 }
            }
          },
          y: {
            title: { display: true, text: 'Energy (kWh)', color: theme.legendColor },
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 11 } }
          }
        }
      }
    });
  }

  /**
   * 2. 24-Lag ACF with 95% Confidence Corridor (+/- 0.1512)
   */
  renderAcf(acfDetails, showNoiseLags = true) {
    this.lastAcfArgs = [acfDetails, showNoiseLags];
    const ctx = document.getElementById('acfChart');
    if (!ctx) return;

    const theme = this.getThemeColors();
    if (this.acfChart) {
      this.acfChart.destroy();
    }

    const lags = acfDetails.lags || Array.from({ length: 25 }, (_, i) => i);
    const rawAcf = acfDetails.raw_acf || [];
    const filteredAcf = acfDetails.filtered_acf || [];
    const threshold = acfDetails.threshold || (1.96 / Math.sqrt(168));

    const barData = showNoiseLags ? rawAcf : filteredAcf;

    const backgroundColors = barData.map((val, idx) => {
      if (idx === 0) return 'rgba(124, 58, 237, 0.85)'; // Lag 0
      if (Math.abs(val) > threshold) {
        return val >= 0 ? (theme.isLight ? 'rgba(2, 132, 199, 0.9)' : 'rgba(6, 182, 212, 0.9)') : 'rgba(225, 29, 72, 0.85)';
      }
      return theme.noiseBarColor;
    });

    const upperLine = Array(lags.length).fill(threshold);
    const lowerLine = Array(lags.length).fill(-threshold);

    this.acfChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: lags.map(l => `Lag ${l}`),
        datasets: [
          {
            type: 'bar',
            label: 'Autocorrelation (r)',
            data: barData,
            backgroundColor: backgroundColors,
            borderRadius: 4,
            borderSkipped: false
          },
          {
            type: 'line',
            label: '+95% CI (+0.151)',
            data: upperLine,
            borderColor: 'rgba(217, 119, 6, 0.85)',
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false
          },
          {
            type: 'line',
            label: '-95% CI (-0.151)',
            data: lowerLine,
            borderColor: 'rgba(217, 119, 6, 0.85)',
            borderWidth: 1.5,
            borderDash: [5, 4],
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: theme.legendColor, font: { family: 'Plus Jakarta Sans', size: 11 } }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            bodyColor: theme.tooltipBody,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              afterLabel: function(ctx) {
                if (ctx.datasetIndex === 0) {
                  const val = ctx.raw;
                  const sig = Math.abs(val) > threshold ? 'Significant (Retained)' : 'Noise (Cut to 0.0 in K-Means)';
                  return `Status: ${sig}`;
                }
                return '';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            min: -0.6,
            max: 1.05,
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 11 } }
          }
        }
      }
    });
  }

  /**
   * 3. K-Means Centroid Signatures Comparison (Lags 1 to 24)
   */
  renderCentroids(clusterProfiles) {
    this.lastCentroidsArgs = [clusterProfiles];
    const ctx = document.getElementById('centroidsChart');
    if (!ctx || !clusterProfiles || clusterProfiles.length === 0) return;

    const theme = this.getThemeColors();
    if (this.centroidsChart) {
      this.centroidsChart.destroy();
    }

    const lags = Array.from({ length: 24 }, (_, i) => `Lag ${i + 1}`);

    const datasets = clusterProfiles.map((p, idx) => {
      const centroidValues = p.centroid_acf.length === 25 ? p.centroid_acf.slice(1) : p.centroid_acf;
      const color = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];

      return {
        label: `C${p.cluster_id}: ${p.persona.substring(0, 18)}... (${p.percentage}%)`,
        data: centroidValues,
        borderColor: color,
        backgroundColor: color,
        borderWidth: 2,
        tension: 0.3,
        pointRadius: 2,
        pointHoverRadius: 6
      };
    });

    this.centroidsChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: lags,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: {
              boxWidth: 12,
              color: theme.legendColor,
              font: { family: 'Plus Jakarta Sans', size: 10 }
            }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            borderColor: theme.tooltipBorder,
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 11 } }
          }
        }
      }
    });
  }

  isolateCentroid(clusterId) {
    if (!this.centroidsChart) return;
    this.centroidsChart.data.datasets.forEach((ds, idx) => {
      if (clusterId === 'all' || idx === clusterId) {
        ds.hidden = false;
      } else {
        ds.hidden = true;
      }
    });
    this.centroidsChart.update();
  }

  /**
   * 4. Consumer Segment Proportions Chart
   */
  renderDistribution(clusterProfiles) {
    this.lastDistributionArgs = [clusterProfiles];
    const ctx = document.getElementById('distributionChart');
    if (!ctx || !clusterProfiles) return;

    const theme = this.getThemeColors();
    if (this.distributionChart) {
      this.distributionChart.destroy();
    }

    const labels = clusterProfiles.map(p => `Cluster ${p.cluster_id}: ${p.persona}`);
    const data = clusterProfiles.map(p => p.size);
    const bgColors = clusterProfiles.map((_, i) => CLUSTER_COLORS[i % CLUSTER_COLORS.length]);

    this.distributionChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: data,
          backgroundColor: bgColors,
          borderWidth: 2,
          borderColor: theme.doughnutBorder
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '68%',
        plugins: {
          legend: {
            display: false
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              label: function(ctx) {
                const val = ctx.raw;
                const total = ctx.chart.data.datasets[0].data.reduce((a, b) => a + b, 0);
                const pct = ((val / total) * 100).toFixed(1);
                return ` ${ctx.label}: ${val} homes (${pct}%)`;
              }
            }
          }
        }
      }
    });
  }

  /**
   * 5. Elbow & Silhouette Optimization Curve
   */
  renderElbow(elbowData) {
    this.lastElbowArgs = [elbowData];
    const ctx = document.getElementById('elbowChart');
    if (!ctx || !elbowData) return;

    const theme = this.getThemeColors();
    if (this.elbowChart) {
      this.elbowChart.destroy();
    }

    const kRange = elbowData.k_range;
    const inertias = elbowData.inertias;
    const silhouettes = elbowData.silhouettes;
    const dbis = elbowData.davies_bouldin;

    this.elbowChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: kRange.map(k => `k=${k}`),
        datasets: [
          {
            label: 'Inertia (WCSS)',
            data: inertias,
            borderColor: theme.isLight ? '#0284c7' : '#06b6d4',
            backgroundColor: theme.isLight ? '#0284c7' : '#06b6d4',
            borderWidth: 2,
            yAxisID: 'yInertia',
            tension: 0.2,
            pointRadius: 4
          },
          {
            label: 'Silhouette Score',
            data: silhouettes,
            borderColor: theme.isLight ? '#059669' : '#10b981',
            backgroundColor: theme.isLight ? '#059669' : '#10b981',
            borderWidth: 2,
            yAxisID: 'yMetrics',
            tension: 0.2,
            pointRadius: 4
          },
          {
            label: 'Davies-Bouldin Index (DBI)',
            data: dbis,
            borderColor: theme.isLight ? '#d97706' : '#f59e0b',
            backgroundColor: theme.isLight ? '#d97706' : '#f59e0b',
            borderWidth: 2,
            yAxisID: 'yMetrics',
            borderDash: [4, 4],
            tension: 0.2,
            pointRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: theme.legendColor, font: { family: 'Plus Jakarta Sans', size: 11 } }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            borderColor: theme.tooltipBorder,
            borderWidth: 1
          }
        },
        scales: {
          x: {
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          yInertia: {
            type: 'linear',
            position: 'left',
            title: { display: true, text: 'Inertia (WCSS)', color: theme.isLight ? '#0284c7' : '#06b6d4' },
            grid: { color: theme.gridColor },
            ticks: { color: theme.isLight ? '#0284c7' : '#06b6d4', font: { family: 'JetBrains Mono', size: 10 } }
          },
          yMetrics: {
            type: 'linear',
            position: 'right',
            title: { display: true, text: 'Silhouette / DBI Score', color: theme.isLight ? '#059669' : '#10b981' },
            grid: { drawOnChartArea: false },
            ticks: { color: theme.isLight ? '#059669' : '#10b981', font: { family: 'JetBrains Mono', size: 10 } }
          }
        }
      }
    });
  }

  /**
   * 6. 2D PCA Latent Space Scatter Chart
   */
  renderPca(assignments, k = 12) {
    this.lastPcaArgs = [assignments, k];
    const ctx = document.getElementById('pcaChart');
    if (!ctx || !assignments) return;

    const theme = this.getThemeColors();
    if (this.pcaChart) {
      this.pcaChart.destroy();
    }

    const clusterPoints = {};
    assignments.forEach(pt => {
      const c = pt.cluster;
      if (!clusterPoints[c]) clusterPoints[c] = [];
      clusterPoints[c].push({ x: pt.pca_x, y: pt.pca_y, meter_id: pt.meter_id });
    });

    const datasets = Object.keys(clusterPoints).map(cStr => {
      const cId = parseInt(cStr, 10);
      const color = CLUSTER_COLORS[cId % CLUSTER_COLORS.length];
      return {
        label: `Cluster ${cId}`,
        data: clusterPoints[cId],
        backgroundColor: color,
        borderColor: color,
        pointRadius: 4,
        pointHoverRadius: 7,
        borderWidth: 1
      };
    });

    this.pcaChart = new Chart(ctx, {
      type: 'scatter',
      data: { datasets: datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { boxWidth: 8, color: theme.legendColor, font: { family: 'Plus Jakarta Sans', size: 10 } }
          },
          tooltip: {
            backgroundColor: theme.tooltipBg,
            titleColor: theme.tooltipTitle,
            borderColor: theme.tooltipBorder,
            borderWidth: 1,
            callbacks: {
              label: function(ctx) {
                const pt = ctx.raw;
                return `Meter: ${pt.meter_id} (PC1: ${pt.x}, PC2: ${pt.y})`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Principal Component 1', color: theme.legendColor },
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 10 } }
          },
          y: {
            title: { display: true, text: 'Principal Component 2', color: theme.legendColor },
            grid: { color: theme.gridColor },
            ticks: { color: theme.tickColor, font: { family: 'JetBrains Mono', size: 10 } }
          }
        }
      }
    });
  }
}

window.dashboardCharts = new DashboardCharts();
