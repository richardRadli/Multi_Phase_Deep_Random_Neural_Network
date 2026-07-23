import { useState, useEffect } from 'react';
import { ArrowLeft, Play, Square, Loader2, Image as ImageIcon, AlertCircle, ZoomIn, X } from 'lucide-react';

interface HelmWorkspaceProps {
  darkMode: boolean;
  onBack: () => void;
}

interface HelmResult {
  status: string;
  dataset_name: string;
  method: string;
  output_file: string | null;
  metrics?: {
    train_accuracy: number;
    test_accuracy: number;
    train_precision: number;
    test_precision: number;
    train_recall: number;
    test_recall: number;
    train_f1_score: number;
    test_f1_score: number;
    training_time: number;
  };
}

interface ProgressInfo {
  status?: string;
  telemetry?: {
    current_cycle?: number;
    total_cycles?: number;
    progress_percent?: number;
  };
}

export default function HelmWorkspace({ darkMode, onBack }: HelmWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');

  // HELM Config specifikus paraméterek
  const [numberOfTests, setNumberOfTests] = useState<number>(20);
  const [seed, setSeed] = useState<boolean>(true);
  const [penalty, setPenalty] = useState<number | null>(null);
  const [scalingFactor, setScalingFactor] = useState<number | null>(null);

  // Böngésző memóriából való visszaolvasás (HELM specifikus kulcsok)
  const savedTaskId = localStorage.getItem('helm_active_task_id');
  const savedStatus = savedTaskId ? 'running' : 'idle';
  const savedProgressPercent = Number(localStorage.getItem('helm_progress_percent')) || 0;
  const savedProgressMsg = localStorage.getItem('helm_progress_msg') || '';

  const savedCompletedTestsCount = Number(localStorage.getItem('helm_completed_tests_count')) || 1;
  const [completedTestsCount, setCompletedTestsCount] = useState<number>(savedCompletedTestsCount);

  // Állapotváltozók
  const [taskId, setTaskId] = useState<string | null>(savedTaskId);
  const [status, setStatus] = useState<'idle' | 'running' | 'error'>(savedStatus);
  const [progressMsg, setProgressMsg] = useState<string>(savedProgressMsg);
  const [progressPercent, setProgressPercent] = useState<number>(savedProgressPercent);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [results, setResults] = useState<HelmResult | null>(null);

  const [selectedCycle, setSelectedCycle] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'train' | 'test'>('test');
  const [isZoomed, setIsZoomed] = useState<boolean>(false);

  const clearActiveTask = () => {
    setTaskId(null);
    localStorage.removeItem('helm_active_task_id');
    localStorage.removeItem('helm_progress_percent');
    localStorage.removeItem('helm_progress_msg');
  };

  const getErrorMessage = (err: unknown): string => {
    if (err instanceof Error) return err.message;
    return String(err);
  };

  const getPlotUrls = (outputFile: string, cycleIndex: number, runDataset: string) => {
    const cleanPath = outputFile.replace(/\\/g, '/');
    const parts = cleanPath.split('/');
    const filename = parts[parts.length - 1];

    const timestamp = filename.substring(0, 19);
    const storagePrefix = '/app/storage/';
    let relativePath = '';

    if (cleanPath.startsWith(storagePrefix)) {
      relativePath = cleanPath.substring(storagePrefix.length);
    } else {
      const idx = cleanPath.indexOf('/networks/');
      if (idx !== -1) relativePath = cleanPath.substring(idx + 1);
    }

    const relativeFolder = relativePath.substring(0, relativePath.indexOf('/excel/'));

    const trainImgName = `${timestamp}_cycle_${cycleIndex}_${runDataset}_train.jpg`;
    const testImgName = `${timestamp}_cycle_${cycleIndex}_${runDataset}_test.jpg`;

    return {
      train: `http://localhost:8002/static/${relativeFolder}/confusion_matrix/${trainImgName}`,
      test: `http://localhost:8002/static/${relativeFolder}/confusion_matrix/${testImgName}`
    };
  };

  // Datasets lekérése indításkor (Közös Dataset API a :8000-es porton)
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch('http://localhost:8000/datasets');
        if (response.ok) {
          const data = (await response.json()) as { datasets: string[] };
          setAvailableDatasets(data.datasets);
          if (data.datasets.length > 0) {
            setDatasetName(data.datasets[0]);
          }
        }
      } catch (err) {
        console.error('Failed to fetch datasets:', err);
        const fallback = ['connect4', 'isolete', 'mnist'];
        setAvailableDatasets(fallback);
        setDatasetName(fallback[0]);
      }
    };

    void fetchDatasets();
  }, []);

  // Celery Polling a HELM státuszhoz (:8002)
  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8002/nn/helm/status/${taskId}`);
        if (response.ok) {
          const data = (await response.json()) as {
            status: string;
            info: HelmResult | ProgressInfo | string | null
          };

          if (data.status === 'SUCCESS') {
            setStatus('idle');
            setProgressPercent(100);
            setResults(data.info as HelmResult);
            setSelectedCycle(0);

            const runTests = Number(localStorage.getItem('helm_active_tests_count')) || numberOfTests;
            setCompletedTestsCount(runTests);
            localStorage.setItem('helm_completed_tests_count', String(runTests));

            clearActiveTask();
            localStorage.removeItem('helm_active_tests_count');
            clearInterval(interval);
          } else if (data.status === 'FAILURE') {
            setStatus('error');
            setErrorMessage(typeof data.info === 'string' ? data.info : 'Unknown HELM engine error.');
            clearActiveTask();
            localStorage.removeItem('helm_active_tests_count');
            clearInterval(interval);
          } else if (data.status === 'PROGRESS') {
            const progressInfo = data.info as ProgressInfo;
            const newMsg = progressInfo.status || 'Executing hierarchical steps...';
            const newPercent = progressInfo.telemetry?.progress_percent !== undefined
              ? progressInfo.telemetry.progress_percent
              : progressPercent;

            setProgressMsg(newMsg);
            setProgressPercent(newPercent);

            localStorage.setItem('helm_progress_percent', String(newPercent));
            localStorage.setItem('helm_progress_msg', newMsg);
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  const handleStart = async () => {
    setStatus('running');
    setErrorMessage('');
    setResults(null);
    setProgressMsg('Queuing task in Celery...');
    setProgressPercent(0);

    const payload = {
      seed: seed,
      num_tests: numberOfTests,
      penalty: penalty,
      scaling_factor: scalingFactor
    };

    try {
      const queryParams = `dataset_name=${datasetName}`;
      const response = await fetch(`http://localhost:8002/nn/helm/start?${queryParams}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json() as { detail?: string };
        throw new Error(errData.detail || 'Failed to start HELM process.');
      }

      const data = await response.json() as { task_id?: string };
      if (data.task_id) {
        setTaskId(data.task_id);
        localStorage.setItem('helm_active_task_id', data.task_id);
        localStorage.setItem('helm_active_tests_count', String(numberOfTests));
        localStorage.setItem('helm_progress_percent', '0');
        localStorage.setItem('helm_progress_msg', 'Queuing task in Celery...');
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(getErrorMessage(err));
      clearActiveTask();
      localStorage.removeItem('helm_active_tests_count');
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      await fetch(`http://localhost:8002/nn/helm/stop/${taskId}`, { method: 'POST' });
      clearActiveTask();
      localStorage.removeItem('helm_active_tests_count');
      setStatus('idle');
      setProgressMsg('');
      setProgressPercent(0);
    } catch (err) {
      console.error('Failed to send stop signal:', getErrorMessage(err));
    }
  };

  const plotUrls = results && results.output_file
    ? getPlotUrls(results.output_file, selectedCycle, results.dataset_name)
    : null;
  const activeImageUrl = plotUrls ? (activeTab === 'train' ? plotUrls.train : plotUrls.test) : '';

  return (
    <div className="space-y-6">
      {/* CÍMSOR ÉS MÓD JELZŐ */}
      <div className="flex items-center justify-end">
        <span className="text-xs font-mono text-purple-500">HELM Workspace</span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* BAL OSZLOP: PARAMÉTEREK ÉS NAVIGÁCIÓ */}
        <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-5 flex flex-col justify-between h-full shadow-md`}>
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>HELM Parameters</h3>

            <div className="space-y-4">
              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Dataset</label>
                <select
                  value={datasetName}
                  disabled={status === 'running'}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className={`w-full text-sm p-2 rounded-md border ${
                    darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                  } disabled:opacity-50`}
                >
                  {availableDatasets.map((ds) => (
                    <option key={ds} value={ds}>
                      {ds.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>

              <div className={`pt-2 border-t space-y-4 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                {/* Penalty Override */}
                <div>
                  <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>C Penalty (Override)</label>
                  <input
                    type="number"
                    step="0.00001"
                    placeholder="Dataset default if empty"
                    disabled={status === 'running'}
                    value={penalty === null ? '' : penalty}
                    onChange={(e) => setPenalty(e.target.value === '' ? null : parseFloat(e.target.value))}
                    className={`w-full text-sm p-2 rounded-md border ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                    } disabled:opacity-50`}
                  />
                </div>

                {/* Scaling Factor Override */}
                <div>
                  <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Scaling Factor (Override)</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0.0"
                    max="1.0"
                    placeholder="Dataset default if empty"
                    disabled={status === 'running'}
                    value={scalingFactor === null ? '' : scalingFactor}
                    onChange={(e) => setScalingFactor(e.target.value === '' ? null : parseFloat(e.target.value))}
                    className={`w-full text-sm p-2 rounded-md border ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                    } disabled:opacity-50`}
                  />
                </div>

                {/* Tests & Seed */}
                <div className="grid grid-cols-2 gap-3 items-center">
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Tests Count</label>
                    <input
                      type="number"
                      min="1"
                      disabled={status === 'running'}
                      value={numberOfTests}
                      onChange={(e) => setNumberOfTests(parseInt(e.target.value) || 1)}
                      className={`w-full text-sm p-2 rounded-md border ${
                        darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                      } disabled:opacity-50`}
                    />
                  </div>
                  <div className="flex items-center gap-2 mt-4">
                    <input
                      type="checkbox"
                      id="seed"
                      disabled={status === 'running'}
                      checked={seed}
                      onChange={(e) => setSeed(e.target.checked)}
                      className="cursor-pointer disabled:opacity-50"
                    />
                    <label htmlFor="seed" className={`text-xs font-medium cursor-pointer disabled:opacity-50 ${
                      darkMode ? 'text-slate-400' : 'text-slate-600'
                    }`}>Fix Random Seed</label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* GOMBOK: INDÍTÁS ÉS VISSZA A DASHBOARDRA */}
          <div className="pt-4 space-y-2">
            {status === 'running' ? (
              <button
                onClick={handleStop}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Square className="w-4 h-4 fill-white" /> Stop HELM Process
              </button>
            ) : (
              <button
                onClick={handleStart}
                className="w-full py-2.5 bg-purple-600 hover:bg-purple-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Play className="w-4 h-4 fill-white" /> Execute HELM Run
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

        {/* JOBB OSZLOP: EREDMÉNYEK */}
        <div className="lg:col-span-2 flex flex-col h-full space-y-4">
          {status === 'error' && (
            <div className={`p-4 rounded-xl border flex items-start gap-3 shrink-0 ${
              darkMode ? 'bg-rose-950/30 border-rose-900/50 text-rose-400' : 'bg-rose-50 border-rose-200 text-rose-700'
            }`}>
              <AlertCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-sm">Process Failed</h4>
                <p className="text-xs mt-1">{errorMessage}</p>
              </div>
            </div>
          )}

          {status === 'running' && (
            <div className={`p-4 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-2 shrink-0 shadow-sm`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-purple-500" />
                  <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>HELM is processing hierarchy...</h4>
                </div>
                <span className="text-sm font-mono font-bold text-purple-500">{progressPercent}%</span>
              </div>
              <div className={`w-full rounded-full h-2 overflow-hidden ${darkMode ? 'bg-slate-800' : 'bg-slate-200'}`}>
                <div
                  className="bg-purple-500 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progressPercent}%` }}
                ></div>
              </div>
              <p className={`text-xs italic ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>{progressMsg}</p>
            </div>
          )}

          <div className={`flex-1 p-6 rounded-2xl border flex flex-col justify-center items-center ${
            darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
          } shadow-md overflow-hidden`}>
            {results && plotUrls ? (
              <div className="flex flex-col h-full w-full justify-between space-y-4">
                <div className={`flex flex-col sm:flex-row justify-between items-center gap-4 border-b pb-3 shrink-0 ${
                  darkMode ? 'border-slate-800/40' : 'border-slate-200'
                }`}>
                  <h4 className="font-bold text-sm text-purple-500">Task Completed Successfully!</h4>

                  {/* FÜLEK: TEST SET / TRAIN SET */}
                  <div className={`flex p-1 rounded-lg border text-[11px] ${
                    darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
                  }`}>
                    <button
                      onClick={() => setActiveTab('test')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                        activeTab === 'test' 
                          ? 'bg-purple-600 text-white font-bold' 
                          : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Test Set
                    </button>
                    <button
                      onClick={() => setActiveTab('train')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                        activeTab === 'train' 
                          ? 'bg-purple-600 text-white font-bold' 
                          : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Train Set
                    </button>
                  </div>

                  {completedTestsCount > 1 && (
                    <div className="flex gap-1 items-center">
                      <span className={`text-[11px] ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Cycle:</span>
                      {completedTestsCount <= 8 ? (
                        <div className="flex gap-1">
                          {Array.from({ length: completedTestsCount }).map((_, i) => (
                            <button
                              key={i}
                              onClick={() => setSelectedCycle(i)}
                              className={`px-2 py-0.5 text-xs font-mono rounded cursor-pointer transition-all ${
                                selectedCycle === i 
                                  ? 'bg-purple-600 text-white font-bold' 
                                  : darkMode 
                                    ? 'bg-slate-800 text-slate-400 hover:text-white' 
                                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                              }`}
                            >
                              #{i}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <select
                          value={selectedCycle}
                          onChange={(e) => setSelectedCycle(Number(e.target.value))}
                          className={`text-xs font-mono p-1 rounded border cursor-pointer transition-all ${
                            darkMode 
                              ? 'bg-slate-800 border-slate-700 text-slate-200' 
                              : 'bg-white border-slate-300 text-slate-800'
                          }`}
                        >
                          {Array.from({ length: completedTestsCount }).map((_, i) => (
                            <option key={i} value={i}>
                              Cycle #{i}
                            </option>
                          ))}
                        </select>
                      )}
                    </div>
                  )}
                </div>

                {results.metrics && (
                  <div className="space-y-4 shrink-0 animate-fadeIn">
                    {activeTab === 'train' && (
                      <div className={`p-4 rounded-xl border flex items-center justify-between ${
                        darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'
                      } shadow-sm`}>
                        <span className={`text-[11px] font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                          Average Training Time
                        </span>
                        <span className="text-sm font-mono font-bold text-amber-500">
                          {results.metrics.training_time ? `${results.metrics.training_time.toFixed(4)} seconds` : '0.0000 seconds'}
                        </span>
                      </div>
                    )}

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 w-full">
                      <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'} shadow-sm`}>
                        <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                          {activeTab === 'train' ? 'Train Accuracy' : 'Test Accuracy'}
                        </span>
                        <span className="text-xl font-mono font-bold text-purple-500">
                          {((activeTab === 'train' ?  results.metrics.train_accuracy : results.metrics.test_accuracy) * 100).toFixed(2)}%
                        </span>
                      </div>

                      <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'} shadow-sm`}>
                        <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                          {activeTab === 'train' ? 'Train Precision' : 'Test Precision'}
                        </span>
                        <span className="text-xl font-mono font-bold text-sky-500">
                          {(activeTab === 'train' ? results.metrics.train_precision : results.metrics.test_precision).toFixed(4)}
                        </span>
                      </div>

                      <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'} shadow-sm`}>
                        <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                          {activeTab === 'train' ? 'Train Recall' : 'Test Recall'}
                        </span>
                        <span className="text-xl font-mono font-bold text-amber-500">
                          {(activeTab === 'train' ? results.metrics.train_recall : results.metrics.test_recall).toFixed(4)}
                        </span>
                      </div>

                      <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'} shadow-sm`}>
                        <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                          {activeTab === 'train' ? 'Train F1-Score' : 'Test F1-Score'}
                        </span>
                        <span className="text-xl font-mono font-bold text-indigo-500">
                          {(activeTab === 'train' ? results.metrics.train_f1_score : results.metrics.test_f1_score).toFixed(4)}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                <div className={`flex-1 flex items-center justify-center relative group border rounded-xl overflow-hidden bg-white p-4 shadow-inner min-h-[300px] max-h-[520px] ${
                  darkMode ? 'border-slate-800/40' : 'border-slate-200'
                }`}>
                  <div className="absolute top-4 right-4 bg-slate-900/90 text-white px-3 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none flex items-center gap-1 text-xs font-medium z-10 shadow-md">
                    <ZoomIn className="w-4 h-4 text-purple-400" /> Click to view full screen
                  </div>
                  <img
                    src={activeImageUrl}
                    alt="HELM Hierarchical Confusion Matrix"
                    onClick={() => setIsZoomed(true)}
                    className="w-full h-full object-contain cursor-zoom-in transition-transform duration-300 group-hover:scale-[1.01]"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = '';
                    }}
                  />
                </div>

                {/* JAVÍTVA: Excel elmentett fájl elérési útjának kontrasztja Light mode-ban */}
                <div className={`text-[11px] text-center shrink-0 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Excel Saved to: <code className={`px-2 py-1 rounded font-mono ${
                    darkMode
                      ? 'bg-slate-950 text-purple-400'
                      : 'bg-slate-100 border border-slate-200 text-purple-600'
                  }`}>{results.output_file}</code>
                </div>
              </div>
            ) : (
              <div className="text-center space-y-2 py-12">
                <ImageIcon className={`w-12 h-12 mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
                <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>No Active Plot Data</h4>
                <p className={`text-xs max-w-xs ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                  Run the HELM autoencoder series to generate and display the hierarchical layout matrices.
                </p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* FULLSCREEN ZOOM */}
      {isZoomed && activeImageUrl && (
        <div
          onClick={() => setIsZoomed(false)}
          className="fixed inset-0 bg-slate-950/95 z-50 flex items-center justify-center p-4 md:p-8 transition-opacity duration-300 animate-fadeIn"
        >
          <button
            onClick={() => setIsZoomed(false)}
            className="absolute top-6 right-6 p-2 bg-slate-800 hover:bg-slate-700 text-white rounded-full cursor-pointer transition-all"
          >
            <X className="w-6 h-6" />
          </button>

          <div className="max-w-[95vw] max-h-[90vh] flex flex-col items-center space-y-3" onClick={(e) => e.stopPropagation()}>
            <img
              src={activeImageUrl}
              alt="HELM Fullscreen Confusion Matrix"
              className="w-full h-auto max-h-[80vh] object-contain rounded-xl bg-white p-4 shadow-2xl border border-slate-800"
            />
            <span className="text-xs font-mono text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-full">
              {results ? results.dataset_name.toUpperCase() : ''} • Cycle #{selectedCycle} • {activeTab.toUpperCase()}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}