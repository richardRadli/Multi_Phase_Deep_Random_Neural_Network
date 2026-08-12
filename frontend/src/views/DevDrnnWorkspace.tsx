import { useState, useEffect, useRef } from 'react';
import { ArrowLeft, Play, Square, Loader2, Image as ImageIcon, AlertCircle, ZoomIn, X, CheckCircle } from 'lucide-react';

interface DevDrnnWorkspaceProps {
  darkMode: boolean;
  mode?: 'run' | 'tune';
  onBack: () => void;
}

interface DevDrnnResult {
  status: string;
  dataset_name: string;
  method?: string;
  backend?: string;
  output_file?: string | null;
  best_accuracy?: number;
  best_params?: Record<string, any>;
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
  progress_percent?: number;
  current_trial?: number;
  total_trials?: number;
  best_accuracy_so_far?: number;
  telemetry?: {
    current_cycle?: number;
    total_cycles?: number;
    progress_percent?: number;
  };
}

const formatAccuracy = (val: number | undefined): string => {
  if (typeof val !== 'number') return '0.00%';
  const pct = val <= 1.0 ? val * 100 : val;
  return `${pct.toFixed(2)}%`;
};

export default function DevDrnnWorkspace({ darkMode, mode = 'run', onBack }: DevDrnnWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');
  const [method, setMethod] = useState<'BASE' | 'EXP_ORT' | 'EXP_ORT_C'>('BASE');
  const [activation, setActivation] = useState<string>('LeakyReLU');

  // DevDRNN Execution Config parameters
  const [numberOfTests, setNumberOfTests] = useState<number>(20);
  const [seed, setSeed] = useState<boolean>(true);
  const [sigma, setSigma] = useState<number>(0.1);
  const [penalty, setPenalty] = useState<number | null>(null);

  const [neuronMode, setNeuronMode] = useState<'standard' | 'custom3'>('standard');
  const [numOfLayers, setNumOfLayers] = useState<number>(3);
  const [numOfNeurons, setNumOfNeurons] = useState<number>(100);

  const [layer1Neurons, setLayer1Neurons] = useState<number>(100);
  const [layer2Neurons, setLayer2Neurons] = useState<number>(50);
  const [layer3Neurons, setLayer3Neurons] = useState<number>(25);

  const [decayRate, setDecayRate] = useState<number>(0.5);
  const [rcond, setRcond] = useState<number | null>(null);

  // DevDRNN Tuning Config parameters
  const [backend, setBackend] = useState<'optuna' | 'ray'>('optuna');
  const [nTrials, setNTrials] = useState<number>(25);
  const [rcondMin, setRcondMin] = useState<number>(1e-30);
  const [rcondMax, setRcondMax] = useState<number>(1e-1);
  const [penaltyMin, setPenaltyMin] = useState<number>(0.1);
  const [penaltyMax, setPenaltyMax] = useState<number>(30.0);

  // Layer neuron search ranges
  const [l1Min, setL1Min] = useState<number>(600);
  const [l1Max, setL1Max] = useState<number>(1000);
  const [l2Min, setL2Min] = useState<number>(200);
  const [l2Max, setL2Max] = useState<number>(500);
  const [l3Min, setL3Min] = useState<number>(50);
  const [l3Max, setL3Max] = useState<number>(100);

  // Read state from localStorage
  const savedTaskId = localStorage.getItem(`dev_drnn_active_task_id_${mode}`);
  const savedStatus = savedTaskId ? 'running' : 'idle';
  const savedProgressPercent = Number(localStorage.getItem(`dev_drnn_progress_percent_${mode}`)) || 0;
  const savedProgressMsg = localStorage.getItem(`dev_drnn_progress_msg_${mode}`) || '';

  const savedCompletedTestsCount = Number(localStorage.getItem('dev_drnn_completed_tests_count')) || 1;
  const [completedTestsCount, setCompletedTestsCount] = useState<number>(savedCompletedTestsCount);

  // Runtime states
  const [taskId, setTaskId] = useState<string | null>(savedTaskId);
  const [status, setStatus] = useState<'idle' | 'running' | 'error'>(savedStatus);
  const [progressMsg, setProgressMsg] = useState<string>(savedProgressMsg);
  const [progressPercent, setProgressPercent] = useState<number>(savedProgressPercent);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [results, setResults] = useState<DevDrnnResult | null>(null);

  const [bestAccSoFar, setBestAccSoFar] = useState<number | null>(null);
  const [selectedCycle, setSelectedCycle] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'train' | 'test'>('test');
  const [isZoomed, setIsZoomed] = useState<boolean>(false);
  const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const clearActiveTask = () => {
    setTaskId(null);
    localStorage.removeItem(`dev_drnn_active_task_id_${mode}`);
    localStorage.removeItem(`dev_drnn_progress_percent_${mode}`);
    localStorage.removeItem(`dev_drnn_progress_msg_${mode}`);
  };

  const handleMethodChange = (newMethod: 'BASE' | 'EXP_ORT' | 'EXP_ORT_C') => {
    setMethod(newMethod);
    if (newMethod !== 'EXP_ORT_C') {
      setPenalty(null);
    } else {
      setPenalty(0.01);
    }
  };

  const getErrorMessage = (err: unknown): string => {
    if (err instanceof Error) return err.message;
    return String(err);
  };

  const getPlotUrls = (outputFile: string, cycleIndex: number, runDataset: string, runMethod: string) => {
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
    const trainImgName = `${timestamp}_cycle_${cycleIndex}_${runDataset}_${runMethod}_train.jpg`;
    const testImgName = `${timestamp}_cycle_${cycleIndex}_${runDataset}_${runMethod}_test.jpg`;

    return {
      train: `http://localhost:8003/static/${relativeFolder}/confusion_matrix/${trainImgName}`,
      test: `http://localhost:8003/static/${relativeFolder}/confusion_matrix/${testImgName}`
    };
  };

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
        const fallback = ['connect4', 'mnist'];
        setAvailableDatasets(fallback);
        setDatasetName(fallback[0]);
      }
    };

    void fetchDatasets();
  }, []);

  useEffect(() => {
    if (!taskId) return;

    let statusUrl = `http://localhost:8003/nn/dev_drnn/status/${taskId}`;
    if (mode === 'tune') {
      statusUrl = `http://localhost:8003/nn/dev_drnn/tune/status/${taskId}`;
    }

    pollingRef.current = setInterval(async () => {
      try {
        const response = await fetch(statusUrl);
        if (response.ok) {
          const data = (await response.json()) as {
            status: string;
            info: DevDrnnResult | ProgressInfo | string | null
          };

          if (data.status === 'SUCCESS') {
            setStatus('idle');
            setProgressPercent(100);
            setResults(data.info as DevDrnnResult);
            setSelectedCycle(0);

            if (mode === 'run') {
              const runTests = Number(localStorage.getItem('dev_drnn_active_tests_count')) || numberOfTests;
              setCompletedTestsCount(runTests);
              localStorage.setItem('dev_drnn_completed_tests_count', String(runTests));
            }

            clearActiveTask();
            localStorage.removeItem('dev_drnn_active_tests_count');
            if (pollingRef.current) clearInterval(pollingRef.current);
          } else if (data.status === 'FAILURE') {
            setStatus('error');
            setErrorMessage(typeof data.info === 'string' ? data.info : 'Unknown DevDRNN engine error.');
            clearActiveTask();
            localStorage.removeItem('dev_drnn_active_tests_count');
            if (pollingRef.current) clearInterval(pollingRef.current);
          } else if (data.status === 'PROGRESS') {
            const progressInfo = data.info as ProgressInfo;

            let newPercent = 0;
            let newMsg = '';

            if (mode === 'tune') {
              newPercent = progressInfo.progress_percent !== undefined ? progressInfo.progress_percent : progressPercent;
              newMsg = (progressInfo.current_trial && progressInfo.total_trials)
                ? `Trial ${progressInfo.current_trial} / ${progressInfo.total_trials}`
                : (progressInfo.status || 'Exploring Hyperparameter Space...');

              if (typeof progressInfo.best_accuracy_so_far === 'number') {
                setBestAccSoFar(progressInfo.best_accuracy_so_far);
              }
            } else {
              newMsg = progressInfo.status || 'Calculating phases...';
              newPercent = progressInfo.telemetry?.progress_percent !== undefined
                ? progressInfo.telemetry.progress_percent
                : progressPercent;
            }

            setProgressMsg(newMsg);
            setProgressPercent(newPercent);

            localStorage.setItem(`dev_drnn_progress_percent_${mode}`, String(newPercent));
            localStorage.setItem(`dev_drnn_progress_msg_${mode}`, newMsg);
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1000);

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, [taskId, mode]);

  const handleStart = async () => {
    setStatus('running');
    setErrorMessage('');
    setResults(null);
    setBestAccSoFar(null);
    setProgressMsg('Queuing task in Celery...');
    setProgressPercent(0);

    let startUrl = `http://localhost:8003/nn/dev_drnn/start?dataset_name=${datasetName}&method=${method}&activation=${activation}`;
    let payload: Record<string, unknown> = {
      number_of_tests: numberOfTests,
      seed: seed,
      sigma: method === 'BASE' ? undefined : sigma,
      penalty: penalty,
      rcond: rcond
    };

    if (neuronMode === 'custom3') {
      payload.hidden_neurons = [layer1Neurons, layer2Neurons, layer3Neurons];
    } else {
      payload.num_of_layers = numOfLayers;
      payload.num_of_neurons = numOfNeurons;
      payload.decay_rate = decayRate;
    }

    if (mode === 'tune') {
      startUrl = `http://localhost:8003/nn/dev_drnn/tune?dataset_name=${datasetName}`;
      payload = {
        backend,
        n_trials: nTrials,
        seed: seed,
        method: method,
        activation: activation,
        rcond_min: rcondMin,
        rcond_max: rcondMax,
        penalty_min: penaltyMin,
        penalty_max: penaltyMax,
        l1_min: l1Min,
        l1_max: l1Max,
        l2_min: l2Min,
        l2_max: l2Max,
        l3_min: l3Min,
        l3_max: l3Max
      };
    }

    try {
      const response = await fetch(startUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json() as { detail?: string };
        throw new Error(errData.detail || 'Failed to start DevDRNN process.');
      }

      const data = await response.json() as { task_id?: string };
      if (data.task_id) {
        setTaskId(data.task_id);
        localStorage.setItem(`dev_drnn_active_task_id_${mode}`, data.task_id);
        localStorage.setItem('dev_drnn_active_tests_count', String(numberOfTests));
        localStorage.setItem(`dev_drnn_progress_percent_${mode}`, '0');
        localStorage.setItem(`dev_drnn_progress_msg_${mode}`, 'Queuing task in Celery...');
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(getErrorMessage(err));
      clearActiveTask();
      localStorage.removeItem('dev_drnn_active_tests_count');
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      let stopUrl = `http://localhost:8003/nn/dev_drnn/stop/${taskId}`;
      if (mode === 'tune') {
        stopUrl = `http://localhost:8003/nn/dev_drnn/tune/stop/${taskId}`;
      }
      await fetch(stopUrl, { method: 'POST' });
      clearActiveTask();
      localStorage.removeItem('dev_drnn_active_tests_count');
      setStatus('idle');
      setProgressMsg('');
      setProgressPercent(0);
    } catch (err) {
      console.error('Failed to send stop signal:', getErrorMessage(err));
    }
  };

  const plotUrls = (results && results.output_file && mode === 'run')
    ? getPlotUrls(results.output_file, selectedCycle, results.dataset_name, results.method || method)
    : null;
  const activeImageUrl = plotUrls ? (activeTab === 'train' ? plotUrls.train : plotUrls.test) : '';

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-end">
        <span className="text-xs font-mono text-emerald-500 uppercase tracking-wider">
          DevDRNN {mode === 'tune' ? 'Hyperparameter Tuning Lab' : 'Workspace'}
        </span>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* LEFT COLUMN: PARAMETERS */}
        <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-5 flex flex-col justify-between h-full shadow-md`}>
          <div>
            <h3 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>
              {mode === 'tune' ? 'Tuning Configuration' : 'Model Parameters'}
            </h3>

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

              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Weight Method</label>
                <select
                  value={method}
                  disabled={status === 'running'}
                  onChange={(e) => handleMethodChange(e.target.value as 'BASE' | 'EXP_ORT' | 'EXP_ORT_C')}
                  className={`w-full text-sm p-2 rounded-md border ${
                    darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                  } disabled:opacity-50`}
                >
                  <option value="BASE">BASE</option>
                  <option value="EXP_ORT">EXP_ORT</option>
                  <option value="EXP_ORT_C">EXP_ORT_C</option>
                </select>
              </div>

              <div>
                <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Activation</label>
                <select
                  value={activation}
                  disabled={status === 'running'}
                  onChange={(e) => setActivation(e.target.value)}
                  className={`w-full text-sm p-2 rounded-md border ${
                    darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                  } disabled:opacity-50`}
                >
                  <option value="ReLU">ReLU</option>
                  <option value="LeakyReLU">LeakyReLU</option>
                  <option value="Tanh">Tanh</option>
                  <option value="Sigmoid">Sigmoid</option>
                  <option value="Identity">Identity</option>
                </select>
              </div>

              {mode === 'tune' ? (
                /* TUNING MODE FIELDS */
                <div className={`pt-2 border-t space-y-4 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Search Engine Backend</label>
                    <select
                      value={backend}
                      disabled={status === 'running'}
                      onChange={(e) => setBackend(e.target.value as 'optuna' | 'ray')}
                      className={`w-full text-sm p-2 rounded-md border ${
                        darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                      } disabled:opacity-50`}
                    >
                      <option value="optuna">Optuna (TPE Sampler)</option>
                      <option value="ray">Ray Tune (ASHA Scheduler)</option>
                    </select>
                  </div>

                  <div>
                    <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Total Trials: {nTrials}
                    </label>
                    <input
                      type="range"
                      min="5"
                      max="100"
                      step="5"
                      value={nTrials}
                      disabled={status === 'running'}
                      onChange={(e) => setNTrials(parseInt(e.target.value))}
                      className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                  </div>

                  <div className="space-y-2">
                    <span className={`block text-[11px] font-mono font-semibold ${darkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>
                      Layer Neuron Search Ranges
                    </span>

                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <label className={`block text-[10px] font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 1 Min / Max</label>
                        <div className="flex gap-1">
                          <input
                            type="number"
                            value={l1Min}
                            disabled={status === 'running'}
                            onChange={(e) => setL1Min(parseInt(e.target.value) || 1)}
                            className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            }`}
                          />
                          <input
                            type="number"
                            value={l1Max}
                            disabled={status === 'running'}
                            onChange={(e) => setL1Max(parseInt(e.target.value) || 1)}
                            className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            }`}
                          />
                        </div>
                      </div>

                      <div>
                        <label className={`block text-[10px] font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 2 Min / Max</label>
                        <div className="flex gap-1">
                          <input
                            type="number"
                            value={l2Min}
                            disabled={status === 'running'}
                            onChange={(e) => setL2Min(parseInt(e.target.value) || 1)}
                            className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            }`}
                          />
                          <input
                            type="number"
                            value={l2Max}
                            disabled={status === 'running'}
                            onChange={(e) => setL2Max(parseInt(e.target.value) || 1)}
                            className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            }`}
                          />
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className={`block text-[10px] font-medium ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 3 Min / Max</label>
                      <div className="flex gap-1">
                        <input
                          type="number"
                          value={l3Min}
                          disabled={status === 'running'}
                          onChange={(e) => setL3Min(parseInt(e.target.value) || 1)}
                          className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          }`}
                        />
                        <input
                          type="number"
                          value={l3Max}
                          disabled={status === 'running'}
                          onChange={(e) => setL3Max(parseInt(e.target.value) || 1)}
                          className={`w-1/2 text-xs p-1.5 rounded border font-mono ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          }`}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>rcond Min</label>
                      <input
                        type="number"
                        step="1e-30"
                        value={rcondMin}
                        disabled={status === 'running'}
                        onChange={(e) => setRcondMin(parseFloat(e.target.value) || 1e-30)}
                        className={`w-full text-xs p-2 rounded border font-mono ${
                          darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                        }`}
                      />
                    </div>
                    <div>
                      <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>rcond Max</label>
                      <input
                        type="number"
                        step="0.01"
                        value={rcondMax}
                        disabled={status === 'running'}
                        onChange={(e) => setRcondMax(parseFloat(e.target.value) || 0.1)}
                        className={`w-full text-xs p-2 rounded border font-mono ${
                          darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                        }`}
                      />
                    </div>
                  </div>

                  {method === 'EXP_ORT_C' && (
                    <div className="grid grid-cols-2 gap-3 animate-fadeIn">
                      <div>
                        <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Penalty Term Min</label>
                        <input
                          type="number"
                          step="0.1"
                          value={penaltyMin}
                          disabled={status === 'running'}
                          onChange={(e) => setPenaltyMin(parseFloat(e.target.value) || 0.1)}
                          className={`w-full text-xs p-2 rounded border font-mono ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          }`}
                        />
                      </div>
                      <div>
                        <label className={`block text-[11px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Penalty Term Max</label>
                        <input
                          type="number"
                          step="0.5"
                          value={penaltyMax}
                          disabled={status === 'running'}
                          onChange={(e) => setPenaltyMax(parseFloat(e.target.value) || 30.0)}
                          className={`w-full text-xs p-2 rounded border font-mono ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          }`}
                        />
                      </div>
                    </div>
                  )}

                  <div className="flex items-center gap-2 pt-2">
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
              ) : (
                /* EXECUTION MODE FIELDS */
                <div className={`pt-2 border-t space-y-4 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <div>
                    <label className={`block text-xs font-medium mb-1.5 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Neuron Configuration Mode
                    </label>
                    <div className={`grid grid-cols-2 p-1 rounded-lg border text-xs font-semibold ${
                      darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
                    }`}>
                      <button
                        type="button"
                        disabled={status === 'running'}
                        onClick={() => setNeuronMode('standard')}
                        className={`py-1.5 px-2 rounded-md transition-all cursor-pointer ${
                          neuronMode === 'standard'
                            ? 'bg-emerald-600 text-white shadow-sm'
                            : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Standard (Auto)
                      </button>
                      <button
                        type="button"
                        disabled={status === 'running'}
                        onClick={() => setNeuronMode('custom3')}
                        className={`py-1.5 px-2 rounded-md transition-all cursor-pointer ${
                          neuronMode === 'custom3'
                            ? 'bg-emerald-600 text-white shadow-sm'
                            : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Custom 3-Layer
                      </button>
                    </div>
                  </div>

                  {neuronMode === 'standard' ? (
                    <div className="grid grid-cols-2 gap-3 animate-fadeIn">
                      <div>
                        <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layers</label>
                        <input
                          type="number"
                          min="1"
                          disabled={status === 'running'}
                          value={numOfLayers}
                          onChange={(e) => setNumOfLayers(parseInt(e.target.value) || 1)}
                          className={`w-full text-sm p-2 rounded-md border ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          } disabled:opacity-50`}
                        />
                      </div>
                      <div>
                        <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Total Neurons</label>
                        <input
                          type="number"
                          min="1"
                          disabled={status === 'running'}
                          value={numOfNeurons}
                          onChange={(e) => setNumOfNeurons(parseInt(e.target.value) || 1)}
                          className={`w-full text-sm p-2 rounded-md border ${
                            darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                          } disabled:opacity-50`}
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="space-y-2 animate-fadeIn">
                      <span className={`block text-[11px] font-mono ${darkMode ? 'text-emerald-400' : 'text-emerald-600'}`}>
                        Fixed 3-Layer Architecture Configuration
                      </span>
                      <div className="grid grid-cols-3 gap-2">
                        <div>
                          <label className={`block text-[10px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 1</label>
                          <input
                            type="number"
                            min="1"
                            disabled={status === 'running'}
                            value={layer1Neurons}
                            onChange={(e) => setLayer1Neurons(parseInt(e.target.value) || 1)}
                            className={`w-full text-xs p-2 rounded-md border ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            } disabled:opacity-50`}
                          />
                        </div>
                        <div>
                          <label className={`block text-[10px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 2</label>
                          <input
                            type="number"
                            min="1"
                            disabled={status === 'running'}
                            value={layer2Neurons}
                            onChange={(e) => setLayer2Neurons(parseInt(e.target.value) || 1)}
                            className={`w-full text-xs p-2 rounded-md border ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            } disabled:opacity-50`}
                          />
                        </div>
                        <div>
                          <label className={`block text-[10px] font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Layer 3</label>
                          <input
                            type="number"
                            min="1"
                            disabled={status === 'running'}
                            value={layer3Neurons}
                            onChange={(e) => setLayer3Neurons(parseInt(e.target.value) || 1)}
                            className={`w-full text-xs p-2 rounded-md border ${
                              darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                            } disabled:opacity-50`}
                          />
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Sigma */}
                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Sigma (Deviation)</span>
                      <span className={method === 'BASE' ? (darkMode ? 'text-slate-600' : 'text-slate-400') : 'text-emerald-500 font-semibold'}>
                        {method === 'BASE' ? 'N/A' : sigma}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0.01"
                      max="1.0"
                      step="0.01"
                      value={sigma}
                      disabled={method === 'BASE' || status === 'running'}
                      onChange={(e) => setSigma(parseFloat(e.target.value))}
                      className="w-full cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed accent-emerald-500"
                    />
                  </div>

                  {/* Decay Rate */}
                  <div>
                    <div className="flex justify-between text-xs font-medium mb-1">
                      <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Decay Rate</span>
                      <span className="text-emerald-500 font-semibold">{decayRate}</span>
                    </div>
                    <input
                      type="range"
                      min="0.0"
                      max="2.0"
                      step="0.1"
                      value={decayRate}
                      disabled={status === 'running'}
                      onChange={(e) => setDecayRate(parseFloat(e.target.value))}
                      className="w-full cursor-pointer disabled:opacity-50 accent-emerald-500"
                    />
                  </div>

                  {/* Penalty Term */}
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>L2 Penalty</label>
                    <input
                      type="number"
                      step="0.001"
                      placeholder="e.g. 0.01"
                      disabled={method !== 'EXP_ORT_C' || status === 'running'}
                      value={penalty === null ? '' : penalty}
                      onChange={(e) => setPenalty(e.target.value === '' ? null : parseFloat(e.target.value))}
                      className={`w-full text-sm p-2 rounded-md border disabled:opacity-30 disabled:cursor-not-allowed ${
                        darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-300 text-slate-800'
                      }`}
                    />
                    {method !== 'EXP_ORT_C' && (
                      <span className={`text-[10px] mt-1 block ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>Only allowed for EXP_ORT_C method.</span>
                    )}
                  </div>

                  {/* Moore-Penrose rcond */}
                  <div>
                    <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Moore-Penrose rcond</label>
                    <input
                      type="number"
                      step="0.00001"
                      placeholder="e.g. 1e-5 (Optional)"
                      disabled={status === 'running'}
                      value={rcond === null ? '' : rcond}
                      onChange={(e) => setRcond(e.target.value === '' ? null : parseFloat(e.target.value))}
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
              )}
            </div>
          </div>

          {/* ACTION BUTTONS */}
          <div className="pt-4 space-y-2">
            {status === 'running' ? (
              <button
                onClick={handleStop}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Square className="w-4 h-4 fill-white" /> Stop DevDRNN Process
              </button>
            ) : (
              <button
                onClick={handleStart}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Play className="w-4 h-4 fill-white" /> {mode === 'tune' ? 'Launch Hyperparameter Search' : 'Execute DevDRNN Run'}
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

        {/* RIGHT COLUMN: RESULTS PANEL */}
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
            <div className={`p-4 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-3 shrink-0 shadow-sm`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-emerald-500" />
                  <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                    {mode === 'tune' ? 'Optimizing DevDRNN hyperparameters...' : 'DevDRNN is running...'}
                  </h4>
                </div>
                <span className="text-sm font-mono font-bold text-emerald-500">{progressPercent}%</span>
              </div>
              <div className={`w-full rounded-full h-2 overflow-hidden ${darkMode ? 'bg-slate-800' : 'bg-slate-200'}`}>
                <div
                  className="bg-emerald-500 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progressPercent}%` }}
                ></div>
              </div>
              <p className={`text-xs italic ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>{progressMsg}</p>

              {mode === 'tune' && bestAccSoFar !== null && (
                <div className={`p-3 rounded-xl border text-xs font-mono flex justify-between items-center animate-fadeIn ${
                  darkMode ? 'bg-slate-950/60 border-slate-800 text-slate-300' : 'bg-slate-50 border-slate-200 text-slate-700'
                }`}>
                  <span className="text-slate-500">Current Peak Accuracy:</span>
                  <span className="font-bold text-emerald-400">{formatAccuracy(bestAccSoFar)}</span>
                </div>
              )}
            </div>
          )}

          {/* MAIN RESULTS DISPLAY */}
          <div className={`flex-1 p-6 rounded-2xl border flex flex-col justify-center items-center ${
            darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
          } shadow-md overflow-hidden`}>

            {mode === 'tune' && results && (
              <div className="space-y-6 w-full animate-fadeIn">
                <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <h4 className="font-bold text-base text-emerald-400">Hyperparameter Optimization Finished</h4>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Best Accuracy</span>
                    <span className="text-2xl font-mono font-black text-emerald-400">{formatAccuracy(results.best_accuracy)}</span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Search Backend</span>
                    <span className="text-2xl font-mono font-bold uppercase text-indigo-400">{results.backend ?? 'Optuna'}</span>
                  </div>
                </div>

                {results.best_params && (
                  <div className={`p-5 rounded-xl border space-y-3 ${darkMode ? 'bg-slate-950/50 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <h5 className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-300' : 'text-slate-700'}`}>Discovered Optimal Parameters:</h5>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                      {Object.entries(results.best_params).map(([key, value]) => (
                        <div key={key} className={`p-3 rounded-lg border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'}`}>
                          <span className="block text-[10px] font-mono text-slate-500 uppercase">{key}</span>
                          <span className="text-sm font-mono font-bold text-emerald-400">
                            {Array.isArray(value)
                              ? `[${value.join(', ')}]`
                              : typeof value === 'number'
                                ? value.toExponential(6)
                                : String(value)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {mode === 'run' && results && plotUrls ? (
              <div className="flex flex-col h-full w-full justify-between space-y-4">
                <div className={`flex flex-col sm:flex-row justify-between items-center gap-4 border-b pb-3 shrink-0 ${
                  darkMode ? 'border-slate-800/40' : 'border-slate-200'
                }`}>
                  <h4 className="font-bold text-sm text-emerald-500">Task Completed Successfully!</h4>

                  <div className={`flex p-1 rounded-lg border text-[11px] ${
                    darkMode ? 'bg-slate-950 border-slate-800' : 'bg-slate-100 border-slate-200'
                  }`}>
                    <button
                      onClick={() => setActiveTab('test')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                        activeTab === 'test' 
                          ? 'bg-emerald-600 text-white font-bold' 
                          : darkMode ? 'text-slate-400 hover:text-white' : 'text-slate-600 hover:text-slate-900'
                      }`}
                    >
                      Test Set
                    </button>
                    <button
                      onClick={() => setActiveTab('train')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${
                        activeTab === 'train' 
                          ? 'bg-emerald-600 text-white font-bold' 
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
                                  ? 'bg-emerald-600 text-white font-bold' 
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
                        <span className="text-xl font-mono font-bold text-emerald-500">
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
                    <ZoomIn className="w-4 h-4 text-emerald-400" /> Click to view full screen
                  </div>
                  <img
                    src={activeImageUrl}
                    alt="DevDRNN Phase Confusion Matrix"
                    onClick={() => setIsZoomed(true)}
                    className="w-full h-full object-contain cursor-zoom-in transition-transform duration-300 group-hover:scale-[1.01]"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = '';
                    }}
                  />
                </div>

                <p className={`text-[11px] text-center shrink-0 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                  Excel Saved to: <code className={`px-2 py-1 rounded font-mono ${
                    darkMode
                      ? 'bg-slate-950 text-emerald-400'
                      : 'bg-slate-100 border border-slate-200 text-emerald-600'
                  }`}>{results.output_file}</code>
                </p>
              </div>
            ) : null}

            {status === 'idle' && !results && (
              <div className="text-center space-y-2 py-12">
                <ImageIcon className={`w-12 h-12 mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
                <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>No Active Data</h4>
                <p className={`text-xs max-w-xs ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                  {mode === 'tune' ? 'Configure search limits and launch tuning.' : 'Run the DevDRNN network to generate and display the training phase-plots.'}
                </p>
              </div>
            )}
          </div>

        </div>

      </div>

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
              alt="DevDRNN Fullscreen Confusion Matrix"
              className="w-full h-auto max-h-[80vh] object-contain rounded-xl bg-white p-4 shadow-2xl border border-slate-800"
            />
            <span className="text-xs font-mono text-slate-400 bg-slate-900/80 px-3 py-1.5 rounded-full">
              {results ? results.dataset_name.toUpperCase() : ''} • {results ? results.method : ''} • Cycle #{selectedCycle} • {activeTab.toUpperCase()}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}