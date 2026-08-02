import { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Play, Square, RefreshCw, CheckCircle, AlertCircle, BarChart2, FileSpreadsheet, Settings, Sliders, Cpu, HardDrive, ArrowRight } from 'lucide-react';

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
  mode?: 'single' | 'series';
  dataset_name: string;
  message?: string;
  total_epochs_run?: number;
  execution_time_seconds?: number;
  output_file?: string | null;
  metrics?: FcnnMetrics;
  backend?: string;
  optimizer?: string;
  best_accuracy?: number;
  best_params?: Record<string, any>;
}

interface PollingInfo extends FcnnResults {
  progress_percent?: number;
  current_epoch?: number;
  total_epochs?: number;
  current_trial?: number;
  total_trials?: number;
  best_accuracy_so_far?: number;
  status: string;
}

interface FcnnWorkspaceProps {
  darkMode: boolean;
  mode: 'train' | 'test' | 'tune';
  onBack: () => void;
}

const safeFloat = (val: number | undefined): string => {
  return typeof val === 'number' ? val.toFixed(4) : '0.0000';
};

const formatAccuracy = (val: number | undefined): string => {
  if (typeof val !== 'number') return '0.00%';
  const pct = val <= 1.0 ? val * 100 : val;
  return `${pct.toFixed(2)}%`;
};

export default function FcnnWorkspace({ darkMode, mode, onBack }: FcnnWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');

  const [seed, setSeed] = useState<boolean>(false);
  const [epochs, setEpochs] = useState<number>(mode === 'tune' ? 100 : 1000);
  const [patience, setPatience] = useState<number>(10);
  const [optimizer, setOptimizer] = useState<'adam' | 'sgd'>('adam');
  const [backend, setBackend] = useState<'optuna' | 'ray'>('optuna');
  const [nTrials, setNTrials] = useState<number>(25);

  const [batchSize, setBatchSize] = useState<number>(128);
  const [seriesMode, setSeriesMode] = useState<boolean>(false);
  const [numTests, setNumTests] = useState<number>(20);

  const [lrMin, setLrMin] = useState<number>(0.0004);
  const [lrMax, setLrMax] = useState<number>(0.1);
  const [hiddenNeuronsOpts, setHiddenNeuronsOpts] = useState<number[]>([216, 500, 866, 1000, 2000]);
  const [batchSizeOpts, setBatchSizeOpts] = useState<number[]>([32, 64, 128]);
  const [momentumMin, setMomentumMin] = useState<number>(0.5);
  const [momentumMax, setMomentumMax] = useState<number>(0.99);

  const [activeTab, setActiveTab] = useState<'train' | 'test'>('test');

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

  const [bestAccSoFar, setBestAccSoFar] = useState<number | null>(null);
  const [matrixTimestamp, setMatrixTimestamp] = useState<number>(() => Date.now());
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

    let statusUrl = `http://localhost:8001/nn/fcnn/status/${id}`;
    if (mode === 'test') statusUrl = `http://localhost:8001/nn/fcnn/test/status/${id}`;
    if (mode === 'tune') statusUrl = `http://localhost:8001/nn/fcnn/tune/status/${id}`;

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
          setProgress(Number(data.info.progress_percent) || 0);

          if (mode === 'train') {
            setStatusText(`Epoch: ${data.info.current_epoch ?? 0} / ${data.info.total_epochs ?? 0}`);
          } else if (mode === 'tune') {
            const trialMsg = (data.info.current_trial && data.info.total_trials)
              ? `Trial ${data.info.current_trial} / ${data.info.total_trials}`
              : (data.info.status || 'Exploring Hyperparameter Space...');
            setStatusText(trialMsg);

            if (typeof data.info.best_accuracy_so_far === 'number') {
              setBestAccSoFar(data.info.best_accuracy_so_far);
            }
          } else {
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

  const toggleArrayOption = (list: number[], value: number, setter: (val: number[]) => void) => {
    if (list.includes(value)) {
      if (list.length > 1) setter(list.filter((v) => v !== value));
    } else {
      setter([...list, value].sort((a, b) => a - b));
    }
  };

  const handleStart = async () => {
    setStatus('running');
    setProgress(0);
    setErrorMessage('');
    setResults(null);
    setBestAccSoFar(null);

    try {
      let url = '';
      let body = {};

      if (mode === 'train') {
        url = `http://localhost:8001/nn/fcnn/train?dataset_name=${datasetName}`;
        body = { seed, epochs, patience, optimizer, batch_size: batchSize };
      } else if (mode === 'test') {
        url = `http://localhost:8001/nn/fcnn/test?dataset_name=${datasetName}&batch_size=${batchSize}`;
        body = { seed, series_mode: seriesMode, num_tests: numTests, epochs, optimizer };
      } else {
        url = `http://localhost:8001/nn/fcnn/tune?dataset_name=${datasetName}`;
        body = {
          backend,
          optimizer,
          n_trials: nTrials,
          epochs,
          patience,
          seed,
          lr_min: lrMin,
          lr_max: lrMax,
          hidden_neurons_options: hiddenNeuronsOpts,
          batch_size_options: batchSizeOpts,
          momentum_min: momentumMin,
          momentum_max: momentumMax,
        };
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
      let stopUrl = `http://localhost:8001/nn/fcnn/stop/${taskId}`;
      if (mode === 'test') stopUrl = `http://localhost:8001/nn/fcnn/test/stop/${taskId}`;
      if (mode === 'tune') stopUrl = `http://localhost:8001/nn/fcnn/tune/stop/${taskId}`;

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
        <span className={`text-xs font-mono uppercase tracking-wider ${
          mode === 'train' ? 'text-emerald-500' : mode === 'test' ? 'text-amber-500' : 'text-purple-500'
        }`}>
          FCNN {mode === 'train' ? 'Training Lab' : mode === 'test' ? 'Evaluation Center' : 'Hyperparameter Tuning Lab'}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">

        {/* BAL OSZLOP: PARAMÉTER KÁRTYÁK */}
        <div className="space-y-6">

          {/* 1. KÁRTYA: ENGINE & EXECUTION SETTINGS */}
          <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-5 shadow-md`}>
            <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
              <Settings className={`w-5 h-5 ${
                mode === 'train' ? 'text-emerald-500' : mode === 'test' ? 'text-amber-500' : 'text-purple-500'
              }`} />
              <h3 className={`text-base font-bold ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>
                {mode === 'tune' ? 'Engine & Execution' : 'Hyperparameters'}
              </h3>
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

            {mode === 'tune' && (
              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Search Engine Backend
                </label>
                <select
                  value={backend}
                  disabled={status === 'running'}
                  onChange={(e) => setBackend(e.target.value as 'optuna' | 'ray')}
                  className={`w-full text-sm p-2.5 rounded-md border ${
                    darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                  }`}
                >
                  <option value="optuna">Optuna (TPE Sampler + Pruner)</option>
                  <option value="ray">Ray Tune (ASHA Scheduler)</option>
                </select>
              </div>
            )}

            <div>
              <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Optimizer
              </label>
              <select
                value={optimizer}
                disabled={status === 'running'}
                onChange={(e) => setOptimizer(e.target.value as 'adam' | 'sgd')}
                className={`w-full text-sm p-2.5 rounded-md border ${
                  darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                }`}
              >
                <option value="adam">ADAM</option>
                <option value="sgd">SGD</option>
              </select>
            </div>

            {mode === 'tune' && (
              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Total Trials / Experiments: {nTrials}
                </label>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={nTrials}
                  disabled={status === 'running'}
                  onChange={(e) => setNTrials(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
              </div>
            )}

            <div>
              <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                {mode === 'tune' ? `Max Epochs / Trial: ${epochs}` : `Training Epochs: ${epochs}`}
              </label>
              <input
                type="range"
                min="10"
                max={mode === 'tune' ? 300 : 5000}
                step={mode === 'tune' ? 10 : 50}
                value={epochs}
                disabled={status === 'running'}
                onChange={(e) => setEpochs(parseInt(e.target.value))}
                className={`w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer ${
                  mode === 'train' ? 'accent-emerald-600' : mode === 'test' ? 'accent-amber-500' : 'accent-purple-600'
                }`}
              />
            </div>

            {mode !== 'test' && (
              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Early Stopping Patience: {patience}
                </label>
                <input
                  type="range"
                  min="1"
                  max="50"
                  step="1"
                  value={patience}
                  disabled={status === 'running'}
                  onChange={(e) => setPatience(parseInt(e.target.value))}
                  className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-purple-600"
                />
              </div>
            )}

            <div className={`flex items-center justify-between p-2.5 rounded-lg border ${
              darkMode ? 'border-slate-800/30 bg-slate-950/20' : 'border-slate-200 bg-slate-50'
            }`}>
              <span className={`text-xs font-medium ${darkMode ? 'text-slate-400' : 'text-slate-700'}`}>Fix Random Seed</span>
              <input
                type="checkbox"
                checked={seed}
                disabled={status === 'running'}
                onChange={(e) => setSeed(e.target.checked)}
                className="w-4 h-4 rounded text-purple-600 focus:ring-0 cursor-pointer"
              />
            </div>

            {mode === 'test' && (
              <div className="space-y-4 pt-1">
                <div>
                  <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Evaluation Batch Size</label>
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
              </div>
            )}
          </div>

          {/* 2. KÁRTYA: SEARCH SPACE BOUNDARIES (CSAK TUNING MÓDBAN) */}
          {mode === 'tune' && (
            <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-5 shadow-md animate-fadeIn`}>
              <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                <Sliders className="w-5 h-5 text-purple-500" />
                <h3 className={`text-base font-bold ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>Search Space Boundaries</h3>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>LR Min</label>
                  <input
                    type="number"
                    step="0.0001"
                    value={lrMin}
                    disabled={status === 'running'}
                    onChange={(e) => setLrMin(parseFloat(e.target.value) || 0.0001)}
                    className={`w-full text-xs p-2 rounded border font-mono ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                    }`}
                  />
                </div>
                <div>
                  <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>LR Max</label>
                  <input
                    type="number"
                    step="0.01"
                    value={lrMax}
                    disabled={status === 'running'}
                    onChange={(e) => setLrMax(parseFloat(e.target.value) || 0.1)}
                    className={`w-full text-xs p-2 rounded border font-mono ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                    }`}
                  />
                </div>
              </div>

              <div>
                <label className={`block text-xs font-medium mb-1.5 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Hidden Neurons Search Choices
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {[128, 216, 500, 866, 1000, 2000].map((num) => {
                    const active = hiddenNeuronsOpts.includes(num);
                    return (
                      <button
                        key={num}
                        type="button"
                        disabled={status === 'running'}
                        onClick={() => toggleArrayOption(hiddenNeuronsOpts, num, setHiddenNeuronsOpts)}
                        className={`px-2.5 py-1 text-xs font-mono font-semibold rounded-lg border transition-all cursor-pointer ${
                          active
                            ? 'bg-purple-600 text-white border-purple-500 shadow-sm'
                            : darkMode ? 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white' : 'bg-slate-100 border-slate-200 text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {num}
                      </button>
                    );
                  })}
                </div>
              </div>

              <div>
                <label className={`block text-xs font-medium mb-1.5 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Batch Size Search Choices
                </label>
                <div className="flex flex-wrap gap-1.5">
                  {[16, 32, 64, 128, 256].map((bs) => {
                    const active = batchSizeOpts.includes(bs);
                    return (
                      <button
                        key={bs}
                        type="button"
                        disabled={status === 'running'}
                        onClick={() => toggleArrayOption(batchSizeOpts, bs, setBatchSizeOpts)}
                        className={`px-2.5 py-1 text-xs font-mono font-semibold rounded-lg border transition-all cursor-pointer ${
                          active
                            ? 'bg-purple-600 text-white border-purple-500 shadow-sm'
                            : darkMode ? 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white' : 'bg-slate-100 border-slate-200 text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        {bs}
                      </button>
                    );
                  })}
                </div>
              </div>

              {optimizer === 'sgd' && (
                <div className="grid grid-cols-2 gap-3 pt-1 border-t border-slate-800/40">
                  <div>
                    <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>SGD Momentum Min</label>
                    <input
                      type="number"
                      step="0.05"
                      value={momentumMin}
                      disabled={status === 'running'}
                      onChange={(e) => setMomentumMin(parseFloat(e.target.value) || 0.5)}
                      className={`w-full text-xs p-2 rounded border font-mono ${
                        darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                      }`}
                    />
                  </div>
                  <div>
                    <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>SGD Momentum Max</label>
                    <input
                      type="number"
                      step="0.01"
                      value={momentumMax}
                      disabled={status === 'running'}
                      onChange={(e) => setMomentumMax(parseFloat(e.target.value) || 0.99)}
                      className={`w-full text-xs p-2 rounded border font-mono ${
                        darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                      }`}
                    />
                  </div>
                </div>
              )}
            </div>
          )}

          <div className="space-y-2">
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
                    : mode === 'test'
                    ? 'bg-amber-600 hover:bg-amber-500 shadow-amber-600/10'
                    : 'bg-purple-600 hover:bg-purple-500 shadow-purple-600/10'
                }`}
              >
                <Play className="w-4 h-4 fill-white" /> {
                  mode === 'train' ? 'Fire Up Training Pipeline' : mode === 'test' ? 'Run Weights Evaluation' : 'Launch Hyperparameter Search'
                }
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

          <div className={`p-6 rounded-2xl border flex flex-col justify-start ${
            darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
          } shadow-md min-h-[420px]`}>

            {status === 'running' && (
              <div className="text-center space-y-5 py-12 my-auto">
                <RefreshCw className={`w-12 h-12 mx-auto animate-spin ${
                  mode === 'train' ? 'text-emerald-500' : mode === 'test' ? 'text-amber-500' : 'text-purple-500'
                }`} />
                <div className="space-y-3 max-w-md mx-auto">
                  <div className="flex justify-between text-xs font-mono px-1">
                    <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>{statusText}</span>
                    <span className="font-bold">{progress}%</span>
                  </div>

                  <div className={`w-full rounded-full h-3 overflow-hidden p-[2px] border ${
                    darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-300'
                  }`}>
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        mode === 'train' ? 'bg-emerald-500' : mode === 'test' ? 'bg-amber-500' : 'bg-purple-500'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>

                  {mode === 'tune' && bestAccSoFar !== null && (
                    <div className={`p-3 rounded-xl border text-xs font-mono flex justify-between items-center animate-fadeIn ${
                      darkMode ? 'bg-slate-950/60 border-slate-800 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-700'
                    }`}>
                      <span className="text-slate-500">Current Peak Accuracy:</span>
                      <span className="font-bold text-purple-400">{formatAccuracy(bestAccSoFar)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* TRAIN EREDMÉNYEK */}
            {mode === 'train' && (status === 'success' || status === 'aborted') && results && (
              <div className="space-y-6 animate-fadeIn">
                <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <CheckCircle className={`w-5 h-5 ${status === 'success' ? 'text-emerald-500' : 'text-amber-500'}`} />
                  <h4 className={`font-bold text-base ${status === 'success' ? 'text-emerald-500' : 'text-amber-500'}`}>
                    Training {status === 'success' ? 'Completed Successfully' : 'Gracefully Aborted'}
                  </h4>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Total Epochs Run
                    </span>
                    <span className="text-2xl font-mono font-bold text-emerald-500">{results.total_epochs_run ?? 0}</span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Wall Clock Time
                    </span>
                    <span className="text-2xl font-mono font-bold text-blue-500">{results.execution_time_seconds ?? 0}s</span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Dataset
                    </span>
                    <span className="text-lg font-mono font-bold text-indigo-400 uppercase truncate block mt-1">{datasetName}</span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Optimizer
                    </span>
                    <span className="text-lg font-mono font-bold text-amber-400 uppercase block mt-1">{optimizer}</span>
                  </div>
                </div>

                <div className={`p-5 rounded-xl border space-y-3 ${darkMode ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <div className="flex items-center gap-2">
                    <HardDrive className="w-4 h-4 text-emerald-500" />
                    <h5 className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                      Pipeline Artifacts & Model Checkpoints
                    </h5>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs font-mono">
                    <div className={`p-3 rounded-lg border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                      <span className="text-slate-500 block text-[10px] uppercase">Weights Checkpoint File</span>
                      <span className="font-semibold text-emerald-500 break-all">fcnn_{datasetName}_{optimizer}.pth</span>
                    </div>
                    <div className={`p-3 rounded-lg border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                      <span className="text-slate-500 block text-[10px] uppercase">Model Checkpoint Status</span>
                      <span className="font-semibold text-blue-400">Dumped to Storage Directory</span>
                    </div>
                  </div>
                </div>

                <div className={`p-4 rounded-xl border flex items-center justify-between gap-4 ${
                  darkMode ? 'bg-blue-950/20 border-blue-900/40 text-blue-300' : 'bg-blue-50 border-blue-200 text-blue-800'
                }`}>
                  <div className="text-xs space-y-0.5">
                    <p className="font-bold flex items-center gap-1.5">
                      <ArrowRight className="w-3.5 h-3.5 text-blue-500" /> Next Recommended Action:
                    </p>
                    <p className="opacity-80">Return to Dashboard and open <b>FCNN Testing</b> to evaluate confusion matrix plots and generate Excel metrics reports.</p>
                  </div>
                </div>
              </div>
            )}

            {/* TEST EREDMÉNYEK */}
            {mode === 'test' && status === 'success' && results && (
              <div className="space-y-6">
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
                  </div>
                )}
              </div>
            )}

            {/* TUNING EREDMÉNYEK */}
            {mode === 'tune' && (status === 'success' || status === 'aborted') && results && (
              <div className="space-y-6 animate-fadeIn">
                <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <CheckCircle className={`w-5 h-5 ${status === 'success' ? 'text-purple-500' : 'text-amber-500'}`} />
                  <h4 className={`font-bold text-base ${status === 'success' ? 'text-purple-400' : 'text-amber-500'}`}>
                    Hyperparameter Optimization {status === 'success' ? 'Finished' : 'Manually Aborted'}
                  </h4>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Best Accuracy
                    </span>
                    <span className="text-2xl font-mono font-black text-purple-400">
                      {formatAccuracy(results.best_accuracy)}
                    </span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Engine Backend
                    </span>
                    <span className="text-2xl font-mono font-bold uppercase text-indigo-400">
                      {results.backend ?? 'Optuna'}
                    </span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Optimizer
                    </span>
                    <span className="text-2xl font-mono font-bold uppercase text-amber-400">
                      {results.optimizer ?? 'ADAM'}
                    </span>
                  </div>
                </div>

                {results.best_params && (
                  <div className={`p-5 rounded-xl border space-y-3 ${darkMode ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <h5 className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>
                      Discovered Optimal Configurations:
                    </h5>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      {Object.entries(results.best_params).map(([key, value]) => (
                        <div key={key} className={`p-2.5 rounded-lg border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                          <span className="block text-[10px] font-mono text-slate-500 uppercase">{key}</span>
                          <span className="text-sm font-mono font-bold text-purple-400">
                            {typeof value === 'number' ? value.toFixed(6) : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* IDLE STATE */}
            {status === 'idle' && (
              <div className="text-center space-y-2 py-12 my-auto">
                <Cpu className={`w-12 h-12 mx-auto ${darkMode ? 'text-slate-600' : 'text-slate-400'}`} />
                <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>System Ready & Armed</h4>
                <p className={`text-xs max-w-xs mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                  {mode === 'train'
                    ? 'Adjust training epochs and trigger the network engine to optimize synapses.'
                    : mode === 'test'
                    ? 'Configure batch settings or run iterative Excel evaluation loops directly.'
                    : 'Set boundaries for Learning Rate, Neurons & Batch Sizes to discover peak model parameters.'}
                </p>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}