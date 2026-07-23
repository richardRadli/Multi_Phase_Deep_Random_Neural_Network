import { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Play, Square, RefreshCw, CheckCircle, AlertCircle, BarChart2, FileSpreadsheet, Settings } from 'lucide-react';

interface FcnnMetrics {
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  train_accuracy?: number;
  test_accuracy?: number;
  train_precision?: number;
  test_precision?: number;
  train_recall?: number;
  test_recall?: number;
  train_f1_score?: number;
  test_f1_score?: number;
  training_time?: number;
}

interface FcnnResults {
  status: string;
  mode: 'single' | 'series';
  dataset_name: string;
  message?: string;
  total_epochs_run?: number;
  execution_time_seconds?: number;
  output_file?: string | null;
  metrics?: FcnnMetrics;
}

interface PollingInfo extends FcnnResults {
  progress_percent?: number;
  current_epoch?: number;
  total_epochs?: number;
  status: string;
}

interface FcnnWorkspaceProps {
  darkMode: boolean;
  mode: 'train' | 'test';
  onBack: () => void;
}

const safeFloat = (val: number | undefined): string => {
  return typeof val === 'number' ? val.toFixed(4) : '0.0000';
};

// Intelligens Accuracy formázó (megszorozza 100-al és hozzáadja a % jelet)
const formatAccuracy = (val: number | undefined): string => {
  if (typeof val !== 'number') return '0.00%';
  const pct = val <= 1.0 ? val * 100 : val;
  return `${pct.toFixed(2)}%`;
};

export default function FcnnWorkspace({ darkMode, mode, onBack }: FcnnWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');

  const [seed, setSeed] = useState<boolean>(false);
  const [epochs, setEpochs] = useState<number>(1000);
  const [batchSize, setBatchSize] = useState<number>(128);
  const [seriesMode, setSeriesMode] = useState<boolean>(false);
  const [numTests, setNumTests] = useState<number>(20);

  // Fül váltáshoz (Train / Test)
  const [activeTab, setActiveTab] = useState<'train' | 'test'>('test');

  // Állapotok betöltése LocalStorage-ból
  const [taskId, setTaskId] = useState<string | null>(() =>
    localStorage.getItem(`fcnn_task_id_${mode}`)
  );
  const [status, setStatus] = useState<'idle' | 'running' | 'success' | 'error' | 'aborted'>(() =>
    (localStorage.getItem(`fcnn_status_${mode}`) as 'idle' | 'running' | 'success' | 'error' | 'aborted') || 'idle'
  );
  const [progress, setProgress] = useState<number>(() =>
    Number(localStorage.getItem(`fcnn_progress_${mode}`)) || 0
  );
  const [statusText, setStatusText] = useState<string>(() =>
    localStorage.getItem(`fcnn_status_text_${mode}`) || ''
  );
  const [errorMessage, setErrorMessage] = useState<string>(() =>
    localStorage.getItem(`fcnn_error_${mode}`) || ''
  );
  const [results, setResults] = useState<FcnnResults | null>(() => {
    const saved = localStorage.getItem(`fcnn_results_${mode}`);
    return saved ? JSON.parse(saved) as FcnnResults : null;
  });

  const [matrixTimestamp, setMatrixTimestamp] = useState<number>(() => Date.now());

  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // LocalStorage szinkronizációk
  useEffect(() => {
    if (taskId) localStorage.setItem(`fcnn_task_id_${mode}`, taskId);
    else localStorage.removeItem(`fcnn_task_id_${mode}`);
  }, [taskId, mode]);

  useEffect(() => {
    localStorage.setItem(`fcnn_status_${mode}`, status);
  }, [status, mode]);

  useEffect(() => {
    localStorage.setItem(`fcnn_progress_${mode}`, String(progress));
  }, [progress, mode]);

  useEffect(() => {
    localStorage.setItem(`fcnn_status_text_${mode}`, statusText);
  }, [statusText, mode]);

  useEffect(() => {
    localStorage.setItem(`fcnn_error_${mode}`, errorMessage);
  }, [errorMessage, mode]);

  useEffect(() => {
    if (results) localStorage.setItem(`fcnn_results_${mode}`, JSON.stringify(results));
    else localStorage.removeItem(`fcnn_results_${mode}`);
  }, [results, mode]);

  const startPolling = (id: string) => {
    if (pollingRef.current) clearInterval(pollingRef.current);

    const statusUrl = mode === 'train'
      ? `http://localhost:8001/nn/fcnn/status/${id}`
      : `http://localhost:8001/nn/fcnn/test/status/${id}`;

    pollingRef.current = setInterval(async () => {
      try {
        const res = await fetch(statusUrl);
        if (!res.ok) return;

        const data = await res.json() as {
          status: string;
          info: PollingInfo;
        };

        if (data.status === 'PROGRESS' && data.info) {
          setStatus('running');
          if (mode === 'train') {
            setProgress(Number(data.info.progress_percent) || 0);
            setStatusText(`Epoch: ${data.info.current_epoch ?? 0} / ${data.info.total_epochs ?? 0}`);
          } else {
            setProgress(Number(data.info.progress_percent) || 0);
            setStatusText(String(data.info.status) || `Processing cycles...`);
          }
        } else if (data.status === 'SUCCESS') {
          if (pollingRef.current) clearInterval(pollingRef.current);
          setResults(data.info);
          setStatus('success');
          setProgress(100);
          setMatrixTimestamp(Date.now());
        } else if (data.status === 'ABORTED' || data.status === 'REVOKED') {
          if (pollingRef.current) clearInterval(pollingRef.current);
          setResults(data.info);
          setStatus('aborted');
        } else if (data.status === 'FAILURE') {
          if (pollingRef.current) clearInterval(pollingRef.current);
          setErrorMessage(String(data.info) || 'Task failed execution.');
          setStatus('error');
        }
      } catch (err) {
        console.error('Polling network error:', err);
      }
    }, 1000);
  };

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch('http://localhost:8000/datasets');
        if (response.ok) {
          const data = await response.json() as { datasets: string[] };
          setAvailableDatasets(data.datasets);
          if (data.datasets.length > 0) setDatasetName(data.datasets[0]);
        }
      } catch (err) {
        console.error('Failed to load datasets, using fallback array:', err);
        const fallback = ['connect4', 'mnist', 'letter'];
        setAvailableDatasets(fallback);
        setDatasetName(fallback[0]);
      }
    };
    void fetchDatasets();
  }, []);

  useEffect(() => {
    if (status === 'running' && taskId) {
      startPolling(taskId);
    }
    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const handleStart = async () => {
    setStatus('running');
    setProgress(0);
    setErrorMessage('');
    setResults(null);

    try {
      let url = '';
      let body = {};

      if (mode === 'train') {
        url = `http://localhost:8001/nn/fcnn/train?dataset_name=${datasetName}`;
        body = { seed, epochs };
      } else {
        url = `http://localhost:8001/nn/fcnn/test?dataset_name=${datasetName}&batch_size=${batchSize}`;
        body = { seed, series_mode: seriesMode, num_tests: numTests, epochs: epochs };
      }

      const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errData = await response.json() as { detail?: string };
        throw new Error(errData.detail || 'Failed to trigger FCNN task.');
      }

      const data = await response.json() as { task_id: string };
      setTaskId(data.task_id);
      startPolling(data.task_id);
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : String(err));
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      const stopUrl = mode === 'train'
        ? `http://localhost:8001/nn/fcnn/stop/${taskId}`
        : `http://localhost:8001/nn/fcnn/test/stop/${taskId}`;

      await fetch(stopUrl, { method: 'POST' });
    } catch (err) {
      console.error('Failed to send abort signal:', err);
    }
  };

  const getDisplayMetrics = () => {
    if (!results?.metrics) return { accuracy: 0, precision: 0, recall: 0, f1: 0 };
    const m = results.metrics;

    if (activeTab === 'train') {
      return {
        accuracy: m.train_accuracy ?? m.accuracy ?? 0,
        precision: m.train_precision ?? m.precision ?? 0,
        recall: m.train_recall ?? m.recall ?? 0,
        f1: m.train_f1_score ?? m.f1_score ?? 0,
      };
    } else {
      return {
        accuracy: m.test_accuracy ?? m.accuracy ?? 0,
        precision: m.test_precision ?? m.precision ?? 0,
        recall: m.test_recall ?? m.recall ?? 0,
        f1: m.test_f1_score ?? m.f1_score ?? 0,
      };
    }
  };

  const currentMetrics = getDisplayMetrics();

  return (
    <div className="space-y-6">
      {/* CÍMSOR ÉS MÓD JELZŐ */}
      <div className="flex items-center justify-end">
        <span className={`text-xs font-mono uppercase tracking-wider ${mode === 'train' ? 'text-emerald-500' : 'text-amber-500'}`}>
          FCNN {mode === 'train' ? 'Training Lab' : 'Evaluation Center'}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* BAL OSZLOP: PARAMÉTEREK ÉS NAVIGÁCIÓ */}
        <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} flex flex-col justify-between shadow-md`}>
          <div className="space-y-5">
            <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
              <Settings className={`w-5 h-5 ${mode === 'train' ? 'text-emerald-500' : 'text-amber-500'}`} />
              <h3 className={`text-lg font-semibold ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>Hyperparameters</h3>
            </div>

            <div>
              <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Target Dataset</label>
              <select
                value={datasetName}
                disabled={status === 'running'}
                onChange={(e) => setDatasetName(e.target.value)}
                className={`w-full text-sm p-2.5 rounded-md border ${
                  darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                }`}
              >
                {availableDatasets.map((ds) => (
                  <option key={ds} value={ds}>{ds.toUpperCase()}</option>
                ))}
              </select>
            </div>

            <div className={`flex items-center justify-between p-2.5 rounded-lg border ${
              darkMode ? 'border-slate-800/30 bg-slate-950/20' : 'border-slate-200 bg-slate-50'
            }`}>
              <span className={`text-xs font-medium ${darkMode ? 'text-slate-400' : 'text-slate-700'}`}>Fix Random Seed (42)</span>
              <input
                type="checkbox"
                checked={seed}
                disabled={status === 'running'}
                onChange={(e) => setSeed(e.target.checked)}
                className="w-4 h-4 rounded text-blue-600 focus:ring-0 cursor-pointer"
              />
            </div>

            {mode === 'train' && (
              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Training Epochs: {epochs}
                </label>
                <input
                  type="range"
                  min="10"
                  max="5000"
                  step="50"
                  value={epochs}
                  disabled={status === 'running'}
                  onChange={(e) => setEpochs(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                />
              </div>
            )}

            {mode === 'test' && (
              <div className="space-y-4 pt-1">
                <div>
                  <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    Evaluation Batch Size
                  </label>
                  <select
                    value={batchSize}
                    disabled={status === 'running'}
                    onChange={(e) => setBatchSize(parseInt(e.target.value))}
                    className={`w-full text-sm p-2.5 rounded-md border ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                    }`}
                  >
                    {[16, 32, 64, 128, 256, 512].map((bs) => (
                      <option key={bs} value={bs}>{bs}</option>
                    ))}
                  </select>
                </div>

                <div className={`flex items-center justify-between p-2.5 rounded-lg border ${
                  darkMode ? 'border-slate-800/30 bg-slate-950/20' : 'border-slate-200 bg-slate-50'
                }`}>
                  <span className={`text-xs font-medium ${darkMode ? 'text-slate-400' : 'text-slate-700'}`}>Series Mode</span>
                  <input
                    type="checkbox"
                    checked={seriesMode}
                    disabled={status === 'running'}
                    onChange={(e) => setSeriesMode(e.target.checked)}
                    className="w-4 h-4 rounded text-amber-600 focus:ring-0 cursor-pointer"
                  />
                </div>

                {seriesMode && (
                  <div className="space-y-4 pt-1 animate-fadeIn">
                    <div>
                      <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                        Training Epochs per Cycle: {epochs}
                      </label>
                      <input
                        type="range"
                        min="10"
                        max="5000"
                        step="50"
                        value={epochs}
                        disabled={status === 'running'}
                        onChange={(e) => setEpochs(parseInt(e.target.value) || 10)}
                        className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                      />
                    </div>

                    <div>
                      <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                        Test Cycles: {numTests}
                      </label>
                      <input
                        type="range"
                        min="1"
                        max="50"
                        value={numTests}
                        disabled={status === 'running'}
                        onChange={(e) => setNumTests(parseInt(e.target.value))}
                        className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-amber-500"
                      />
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* GOMBOK: INDÍTÁS ÉS VISSZA A DASHBOARDRA */}
          <div className="pt-4 space-y-2">
            {status === 'running' ? (
              <button
                onClick={handleStop}
                className="w-full py-3 bg-rose-600 hover:bg-rose-500 text-white font-bold rounded-xl text-xs transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md shadow-rose-600/10"
              >
                <Square className="w-4 h-4 fill-white" /> Emergency Graceful Stop
              </button>
            ) : (
              <button
                onClick={handleStart}
                className={`w-full py-3 text-white font-bold rounded-xl text-xs transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md ${
                  mode === 'train' 
                    ? 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/10' 
                    : 'bg-amber-600 hover:bg-amber-500 shadow-amber-600/10'
                }`}
              >
                <Play className="w-4 h-4 fill-white" /> {mode === 'train' ? 'Fire Up Training Pipeline' : 'Run Weights Evaluation'}
              </button>
            )}

            <button
              onClick={onBack}
              className={`w-full py-2.5 rounded-xl border text-xs font-semibold flex items-center justify-center gap-2 cursor-pointer transition-all ${
                darkMode
                  ? 'border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  : 'border-slate-300 text-slate-700 hover:text-slate-900 hover:bg-slate-100'
              }`}
            >
              <ArrowLeft className="w-4 h-4" /> Back to Dashboard
            </button>
          </div>
        </div>

        {/* JOBB OSZLOP: LOGOK ÉS EREDMÉNYEK */}
        <div className="lg:col-span-2 flex flex-col space-y-4">

          {status === 'error' && (
            <div className={`p-4 rounded-xl border flex items-start gap-3 shrink-0 ${
              darkMode ? 'bg-rose-950/30 border-rose-900/50 text-rose-400' : 'bg-rose-50 border-rose-200 text-rose-700'
            }`}>
              <AlertCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-sm">Pipeline Crash Log</h4>
                <p className="text-xs mt-1">{errorMessage}</p>
              </div>
            </div>
          )}

          <div className={`flex-1 p-6 rounded-2xl border flex flex-col justify-center ${
            darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
          } shadow-md`}>

            {status === 'running' && (
              <div className="text-center space-y-5 py-12">
                <RefreshCw className={`w-12 h-12 mx-auto animate-spin ${mode === 'train' ? 'text-emerald-500' : 'text-amber-500'}`} />
                <div className="space-y-2 max-w-md mx-auto">
                  <div className="flex justify-between text-xs font-mono px-1">
                    <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>{statusText}</span>
                    <span className="font-bold">{progress}%</span>
                  </div>
                  <div className={`w-full rounded-full h-3 overflow-hidden p-[2px] border ${
                    darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-300'
                  }`}>
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${mode === 'train' ? 'bg-emerald-500' : 'bg-amber-500'}`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              </div>
            )}

            {mode === 'train' && (status === 'success' || status === 'aborted') && results && (
              <div className="space-y-6">
                <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <CheckCircle className={`w-5 h-5 ${status === 'success' ? 'text-emerald-500' : 'text-amber-500'}`} />
                  <h4 className={`font-bold text-sm ${status === 'success' ? 'text-emerald-500' : 'text-amber-500'}`}>
                    Training {status === 'success' ? 'Completed Successfully' : 'Gracefully Aborted'}
                  </h4>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Total Epochs Run
                    </span>
                    <span className="text-2xl font-mono font-bold text-emerald-500">{results.total_epochs_run ?? 0}</span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Wall Clock Execution Time
                    </span>
                    <span className="text-2xl font-mono font-bold text-blue-500">{results.execution_time_seconds ?? 0}s</span>
                  </div>
                </div>

                <div className={`p-4 rounded-xl border space-y-2 ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className={`text-xs font-bold uppercase tracking-wider block ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    Weights Checkpoint File Status
                  </span>
                  <code className={`block p-2 text-xs font-mono rounded border ${
                    darkMode ? 'bg-slate-950 text-slate-300 border-slate-800' : 'bg-slate-100 text-slate-800 border-slate-200'
                  }`}>
                    Checkpoint compiled & dumped to storage directory. Ready for evaluation.
                  </code>
                </div>
              </div>
            )}

            {mode === 'test' && status === 'success' && results && (
              <div className="space-y-6">
                {/* JAVÍTVA: results.metrics LÉTEZÉSE ESETÉN EGYBŐL MEGJELENÍTJÜK A METRİKÁKAT SINGLE ÉS SERIES MODE-BAN IS! */}
                {results.metrics ? (
                  <div className="space-y-6">
                    <div className={`flex flex-col sm:flex-row items-center justify-between gap-4 border-b pb-3 ${
                      darkMode ? 'border-slate-800/40' : 'border-slate-200'
                    }`}>
                      <div className="flex items-center gap-3">
                        <BarChart2 className="w-5 h-5 text-amber-500" />
                        <h4 className="font-bold text-sm text-amber-500">
                          {results.mode === 'series' ? 'Averaged Series Evaluation Metrics' : 'Evaluation Metrics'}
                        </h4>
                      </div>

                      <div className={`flex p-1 rounded-lg border text-[11px] ${
                        darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
                      }`}>
                        <button
                          onClick={() => setActiveTab('test')}
                          className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                            activeTab === 'test' 
                              ? 'bg-amber-600 text-white font-bold' 
                              : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Test Set
                        </button>
                        <button
                          onClick={() => setActiveTab('train')}
                          className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                            activeTab === 'train' 
                              ? 'bg-amber-600 text-white font-bold' 
                              : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          Train Set
                        </button>
                      </div>
                    </div>

                    <div className="grid grid-cols-4 gap-2 text-center">
                      {[
                        { label: `${activeTab.toUpperCase()} Accuracy`, val: formatAccuracy(currentMetrics.accuracy), color: 'text-emerald-500' },
                        { label: `${activeTab.toUpperCase()} Precision`, val: safeFloat(currentMetrics.precision), color: 'text-blue-500' },
                        { label: `${activeTab.toUpperCase()} Recall`, val: safeFloat(currentMetrics.recall), color: 'text-indigo-500' },
                        { label: `${activeTab.toUpperCase()} F1-Score`, val: safeFloat(currentMetrics.f1), color: 'text-purple-500' }
                      ].map((m) => (
                        <div key={m.label} className={`p-2 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                          <span className={`block text-[9px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>{m.label}</span>
                          <span className={`text-lg font-mono font-bold ${m.color}`}>
                            {m.val}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div className="space-y-2">
                      <span className={`block text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                        Generated {activeTab.toUpperCase()} Confusion Matrix
                      </span>
                      <div className={`w-full rounded-xl border p-2 flex justify-center overflow-hidden h-64 ${
                        darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                      }`}>
                        <img
                          src={`http://localhost:8001/static/networks/fcnn/results_fcnn/${results.dataset_name}/confusion_matrix/${results.dataset_name}_${activeTab}_confusion_matrix.jpg?t=${matrixTimestamp}`}
                          alt={`FCNN ${activeTab} Confusion Matrix`}
                          className="h-full object-contain rounded"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = 'https://via.placeholder.com/400x300?text=Matrix+Plot+Not+Found';
                          }}
                        />
                      </div>
                    </div>

                    {/* JAVÍTVA: HA SERIES MODE-BAN VAN ÉS VAN EXCEL FÁJL, ALUL MEGJELENÍTJÜK AZ ELÉRÉSI ÚTJÁT */}
                    {results.output_file && (
                      <p className={`text-[11px] text-center shrink-0 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                        Excel Report Saved to: <code className={`px-2 py-1 rounded font-mono ${
                          darkMode
                            ? 'bg-slate-950 text-amber-400'
                            : 'bg-slate-100 border border-slate-200 text-amber-600'
                        }`}>{results.output_file}</code>
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="space-y-4 text-center py-6">
                    <FileSpreadsheet className="w-16 h-16 text-emerald-500 mx-auto animate-bounce" />
                    <h4 className="font-bold text-lg text-emerald-500">Excel Test Report Stack Compiled!</h4>
                    <p className={`text-xs max-w-md mx-auto ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>{results.message}</p>
                    <div className={`p-3 rounded border max-w-lg mx-auto ${
                      darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-50 border-slate-200'
                    }`}>
                      <code className="text-[11px] font-mono text-emerald-500 break-all">
                        Artifacts aggregated and saved inside /app/storage/fcnn/excel/
                      </code>
                    </div>
                  </div>
                )}
              </div>
            )}

            {status === 'idle' && (
              <div className="text-center space-y-2 py-12">
                <BarChart2 className={`w-12 h-12 mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
                <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>System Ready & Armed</h4>
                <p className={`text-xs max-w-xs mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                  {mode === 'train'
                    ? 'Adjust the total epochs bar and trigger the network engine to begin optimizing weight synapses.'
                    : 'Configure batch settings or run iterative Excel evaluation loops directly against compiled model checkpoints.'}
                </p>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}