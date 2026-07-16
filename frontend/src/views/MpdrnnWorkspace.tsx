import { useState, useEffect } from 'react';
import { ArrowLeft, Play, Square, Loader2, Image as ImageIcon, AlertCircle, ZoomIn, X } from 'lucide-react';

interface MpdrnnWorkspaceProps {
  darkMode: boolean;
  onBack: () => void;
}

interface MpdrnnResult {
  status: string;
  dataset_name: string;
  method: string;
  output_file: string | null;
}

interface ProgressInfo {
  status?: string;
  telemetry?: {
    current_cycle?: number;
    total_cycles?: number;
    progress_percent?: number;
  };
}

export default function MpdrnnWorkspace({ darkMode, onBack }: MpdrnnWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');
  const [method, setMethod] = useState<'BASE' | 'EXP_ORT' | 'EXP_ORT_C'>('BASE');
  const [activation, setActivation] = useState<string>('LeakyReLU');

  // MPDRNNConfig JSON body paraméterek
  const [numberOfTests, setNumberOfTests] = useState<number>(20);
  const [seed, setSeed] = useState<boolean>(true);
  const [sigma, setSigma] = useState<number>(0.1);
  const [penalty, setPenalty] = useState<number | null>(null);
  const [numOfLayers, setNumOfLayers] = useState<number>(3);
  const [numOfNeurons, setNumOfNeurons] = useState<number>(100);
  const [decayRate, setDecayRate] = useState<number>(0.5);
  const [rcond, setRcond] = useState<number | null>(null);

  // Helyi böngésző memóriából való visszaolvasás az inicializáláshoz
  const savedTaskId = localStorage.getItem('mpdrnn_active_task_id');
  const savedStatus = savedTaskId ? 'running' : 'idle';
  const savedProgressPercent = Number(localStorage.getItem('mpdrnn_progress_percent')) || 0;
  const savedProgressMsg = localStorage.getItem('mpdrnn_progress_msg') || '';

  // --- ÚJ: Eltároljuk a ténylegesen befejezett futás tesztszámait külön ---
  const savedCompletedTestsCount = Number(localStorage.getItem('mpdrnn_completed_tests_count')) || 1;
  const [completedTestsCount, setCompletedTestsCount] = useState<number>(savedCompletedTestsCount);

  // Futási és hálózati állapotok (a mentett adatokkal indítunk!)
  const [taskId, setTaskId] = useState<string | null>(savedTaskId);
  const [status, setStatus] = useState<'idle' | 'running' | 'error'>(savedStatus);
  const [progressMsg, setProgressMsg] = useState<string>(savedProgressMsg);
  const [progressPercent, setProgressPercent] = useState<number>(savedProgressPercent);
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [results, setResults] = useState<MpdrnnResult | null>(null);

  // Vizuális eredményválasztók
  const [selectedCycle, setSelectedCycle] = useState<number>(0);
  const [activeTab, setActiveTab] = useState<'train' | 'test'>('test');

  // Nagyítás állapot
  const [isZoomed, setIsZoomed] = useState<boolean>(false);

  // Helper funkció a localStorage kitakarítására, ha vége a futásnak
  const clearActiveTask = () => {
    setTaskId(null);
    localStorage.removeItem('mpdrnn_active_task_id');
    localStorage.removeItem('mpdrnn_progress_percent');
    localStorage.removeItem('mpdrnn_progress_msg');
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

  // Dinamikus kép URL generátor - a tényleges mérési eredményekből vett nevekkel dolgozik!
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

  // Datasets lekérése indításkor
  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        const response = await fetch('http://localhost:8003/nn/mpdrnn/datasets');
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

  // Celery Polling - megjegyzi a futási részleteket
  useEffect(() => {
    if (!taskId) return;

    const interval = setInterval(async () => {
      try {
        const response = await fetch(`http://localhost:8003/nn/mpdrnn/status/${taskId}`);
        if (response.ok) {
          const data = (await response.json()) as {
            status: string;
            info: MpdrnnResult | ProgressInfo | string | null
          };

          if (data.status === 'SUCCESS') {
            setStatus('idle');
            setProgressPercent(100);
            setResults(data.info as MpdrnnResult);
            setSelectedCycle(0);

            // --- ÚJ: A sikeres futás után rögzítjük az elmentett teszt darabszámot ---
            const runTests = Number(localStorage.getItem('mpdrnn_active_tests_count')) || numberOfTests;
            setCompletedTestsCount(runTests);
            localStorage.setItem('mpdrnn_completed_tests_count', String(runTests));

            clearActiveTask();
            localStorage.removeItem('mpdrnn_active_tests_count');
            clearInterval(interval);
          } else if (data.status === 'FAILURE') {
            setStatus('error');
            setErrorMessage(typeof data.info === 'string' ? data.info : 'Unknown training error.');
            clearActiveTask();
            localStorage.removeItem('mpdrnn_active_tests_count');
            clearInterval(interval);
          } else if (data.status === 'PROGRESS') {
            const progressInfo = data.info as ProgressInfo;
            const newMsg = progressInfo.status || 'Calculating phases...';
            const newPercent = progressInfo.telemetry?.progress_percent !== undefined
              ? progressInfo.telemetry.progress_percent
              : progressPercent;

            setProgressMsg(newMsg);
            setProgressPercent(newPercent);

            localStorage.setItem('mpdrnn_progress_percent', String(newPercent));
            localStorage.setItem('mpdrnn_progress_msg', newMsg);
          }
        }
      } catch (err) {
        console.error('Polling error:', err);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [taskId]);

  // Futás indítása
  const handleStart = async () => {
    setStatus('running');
    setErrorMessage('');
    setResults(null);
    setProgressMsg('Queuing task in Celery...');
    setProgressPercent(0);

    // --- ÚJ: num_tests ÉS number_of_tests IS elmegy a kérésben, így atombiztos a backend fogadás! ---
    const payload = {
      number_of_tests: numberOfTests,
      num_tests: numberOfTests,
      seed: seed,
      sigma: method === 'BASE' ? undefined : sigma,
      penalty: penalty,
      num_of_layers: numOfLayers,
      num_of_neurons: numOfNeurons,
      decay_rate: decayRate,
      rcond: rcond
    };

    try {
      const queryParams = `dataset_name=${datasetName}&method=${method}&activation=${activation}`;
      const response = await fetch(`http://localhost:8003/nn/mpdrnn/start?${queryParams}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errData = await response.json() as { detail?: string };
        throw new Error(errData.detail || 'Failed to start MPDRNN process.');
      }

      const data = await response.json() as { task_id?: string };
      if (data.task_id) {
        setTaskId(data.task_id);
        // Elmentjük az aktív task adatait és darabszámát is
        localStorage.setItem('mpdrnn_active_task_id', data.task_id);
        localStorage.setItem('mpdrnn_active_tests_count', String(numberOfTests));
        localStorage.setItem('mpdrnn_progress_percent', '0');
        localStorage.setItem('mpdrnn_progress_msg', 'Queuing task in Celery...');
      }
    } catch (err) {
      setStatus('error');
      setErrorMessage(getErrorMessage(err));
      clearActiveTask();
      localStorage.removeItem('mpdrnn_active_tests_count');
    }
  };

  const handleStop = async () => {
    if (!taskId) return;
    try {
      await fetch(`http://localhost:8003/nn/mpdrnn/stop/${taskId}`, { method: 'POST' });
      clearActiveTask();
      localStorage.removeItem('mpdrnn_active_tests_count');
      setStatus('idle');
      setProgressMsg('');
      setProgressPercent(0);
    } catch (err) {
      console.error('Failed to send stop signal:', getErrorMessage(err));
    }
  };

  const plotUrls = results && results.output_file
    ? getPlotUrls(results.output_file, selectedCycle, results.dataset_name, results.method)
    : null;
  const activeImageUrl = plotUrls ? (activeTab === 'train' ? plotUrls.train : plotUrls.test) : '';

  return (
    <div className="space-y-6">
      {/* Back to Dashboard */}
      <div className="flex items-center justify-between">
        <button onClick={onBack} className="flex items-center gap-2 text-sm font-medium text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 cursor-pointer">
          <ArrowLeft className="w-4 h-4" /> Back to Dashboard
        </button>
        <span className="text-xs font-mono text-emerald-500">MPDRNN Workspace</span>
      </div>

      {/* ITEMS-STRETCH GRID */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* BAL OSZLOP: PARAMÉTEREK */}
        <div className={`p-6 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-5 flex flex-col justify-between h-full`}>
          <div>
            <h3 className="text-lg font-semibold mb-4">Model Parameters</h3>

            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Dataset</label>
                <select
                  value={datasetName}
                  disabled={status === 'running'}
                  onChange={(e) => setDatasetName(e.target.value)}
                  className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
                >
                  {availableDatasets.map((ds) => (
                    <option key={ds} value={ds}>
                      {ds.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Weight Method</label>
                <select
                  value={method}
                  disabled={status === 'running'}
                  onChange={(e) => handleMethodChange(e.target.value as 'BASE' | 'EXP_ORT' | 'EXP_ORT_C')}
                  className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
                >
                  <option value="BASE">BASE</option>
                  <option value="EXP_ORT">EXP_ORT</option>
                  <option value="EXP_ORT_C">EXP_ORT_C</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Activation</label>
                <select
                  value={activation}
                  disabled={status === 'running'}
                  onChange={(e) => setActivation(e.target.value)}
                  className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
                >
                  <option value="ReLU">ReLU</option>
                  <option value="LeakyReLU">LeakyReLU</option>
                  <option value="Tanh">Tanh</option>
                  <option value="Sigmoid">Sigmoid</option>
                  <option value="Identity">Identity</option>
                </select>
              </div>

              <div className="pt-2 border-t border-slate-800/40 space-y-4">
                {/* Sigma */}
                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-400 mb-1">
                    <span>Sigma (Deviation)</span>
                    <span className={method === 'BASE' ? 'text-slate-600' : 'text-emerald-500 font-semibold'}>
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
                    className="w-full cursor-pointer disabled:opacity-30 disabled:cursor-not-allowed"
                  />
                </div>

                {/* Decay Rate */}
                <div>
                  <div className="flex justify-between text-xs font-medium text-slate-400 mb-1">
                    <span>Decay Rate</span>
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
                    className="w-full cursor-pointer disabled:opacity-50"
                  />
                </div>

                {/* Penalty Term */}
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">L2 Penalty</label>
                  <input
                    type="number"
                    step="0.001"
                    placeholder="e.g. 0.01"
                    disabled={method !== 'EXP_ORT_C' || status === 'running'}
                    value={penalty === null ? '' : penalty}
                    onChange={(e) => setPenalty(e.target.value === '' ? null : parseFloat(e.target.value))}
                    className={`w-full text-sm p-2 rounded-md border disabled:opacity-30 disabled:cursor-not-allowed ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'
                    }`}
                  />
                  {method !== 'EXP_ORT_C' && (
                    <span className="text-[10px] text-slate-500 mt-1 block">Only allowed for EXP_ORT_C method.</span>
                  )}
                </div>

                {/* Moore-Penrose rcond */}
                <div>
                  <label className="block text-xs font-medium text-slate-400 mb-1">Moore-Penrose rcond</label>
                  <input
                    type="number"
                    step="0.00001"
                    placeholder="e.g. 1e-5 (Optional)"
                    disabled={status === 'running'}
                    value={rcond === null ? '' : rcond}
                    onChange={(e) => setRcond(e.target.value === '' ? null : parseFloat(e.target.value))}
                    className={`w-full text-sm p-2 rounded-md border ${
                      darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'
                    } disabled:opacity-50`}
                  />
                </div>

                {/* Neurons & Layers */}
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Layers</label>
                    <input
                      type="number"
                      min="1"
                      disabled={status === 'running'}
                      value={numOfLayers}
                      onChange={(e) => setNumOfLayers(parseInt(e.target.value) || 1)}
                      className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Total Neurons</label>
                    <input
                      type="number"
                      min="1"
                      disabled={status === 'running'}
                      value={numOfNeurons}
                      onChange={(e) => setNumOfNeurons(parseInt(e.target.value) || 1)}
                      className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
                    />
                  </div>
                </div>

                {/* Tests & Seed */}
                <div className="grid grid-cols-2 gap-3 items-center">
                  <div>
                    <label className="block text-xs font-medium text-slate-400 mb-1">Tests Count</label>
                    <input
                      type="number"
                      min="1"
                      disabled={status === 'running'}
                      value={numberOfTests}
                      onChange={(e) => setNumberOfTests(parseInt(e.target.value) || 1)}
                      className={`w-full text-sm p-2 rounded-md border ${darkMode ? 'bg-slate-950 border-slate-700 text-slate-200' : 'bg-white border-slate-200'} disabled:opacity-50`}
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
                    <label htmlFor="seed" className="text-xs font-medium text-slate-400 cursor-pointer disabled:opacity-50">Fix Random Seed</label>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="pt-4">
            {status === 'running' ? (
              <button
                onClick={handleStop}
                className="w-full py-2.5 bg-rose-600 hover:bg-rose-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Square className="w-4 h-4 fill-white" /> Stop MPDRNN Process
              </button>
            ) : (
              <button
                onClick={handleStart}
                className="w-full py-2.5 bg-emerald-600 hover:bg-emerald-500 text-white font-medium rounded-xl text-sm transition-all flex items-center justify-center gap-2 cursor-pointer shadow-sm"
              >
                <Play className="w-4 h-4 fill-white" /> Execute MPDRNN Run
              </button>
            )}
          </div>
        </div>

        {/* JOBB OSZLOP: EREDMÉNYEK PANEL */}
        <div className="lg:col-span-2 flex flex-col h-full space-y-4">

          {status === 'error' && (
            <div className="p-4 rounded-xl bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 flex items-start gap-3 shrink-0">
              <AlertCircle className="w-5 h-5 text-rose-600 dark:text-rose-400 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-sm text-rose-800 dark:text-rose-400">Process Failed</h4>
                <p className="text-xs text-rose-700 dark:text-rose-500 mt-1">{errorMessage}</p>
              </div>
            </div>
          )}

          {status === 'running' && (
            <div className={`p-4 rounded-2xl border ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} space-y-2 shrink-0 shadow-sm`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <Loader2 className="w-5 h-5 animate-spin text-emerald-500" />
                  <h4 className="font-semibold text-sm">MPDRNN is running...</h4>
                </div>
                <span className="text-sm font-mono font-bold text-emerald-500">{progressPercent}%</span>
              </div>
              <div className="w-full bg-slate-200 dark:bg-slate-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-emerald-500 h-2 rounded-full transition-all duration-500 ease-out"
                  style={{ width: `${progressPercent}%` }}
                ></div>
              </div>
              <p className="text-xs text-slate-500 italic">{progressMsg}</p>
            </div>
          )}

          {/* MAIN RESULTS DISPLAY */}
          <div className={`flex-1 p-6 rounded-2xl border flex flex-col justify-center items-center ${darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'} shadow-md overflow-hidden`}>
            {results && plotUrls ? (
              <div className="flex flex-col h-full w-full justify-between space-y-4">

                {/* Fejléc és választók */}
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4 border-b border-slate-800/40 pb-3 shrink-0">
                  <h4 className="font-bold text-sm text-emerald-500">Task Completed Successfully!</h4>

                  {/* Fülek */}
                  <div className="flex bg-slate-950 p-1 rounded-lg border border-slate-800 text-[11px]">
                    <button
                      onClick={() => setActiveTab('test')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${activeTab === 'test' ? 'bg-emerald-600 text-white font-bold' : 'text-slate-400 hover:text-white'}`}
                    >
                      Test Set
                    </button>
                    <button
                      onClick={() => setActiveTab('train')}
                      className={`px-3 py-1 font-semibold rounded cursor-pointer transition-all ${activeTab === 'train' ? 'bg-emerald-600 text-white font-bold' : 'text-slate-400 hover:text-white'}`}
                    >
                      Train Set
                    </button>
                  </div>

                  {/* Ciklus választó - MÓDOSÍTVA: completedTestsCount-ra épül! */}
                  {completedTestsCount > 1 && (
                    <div className="flex gap-1 items-center">
                      <span className="text-[11px] text-slate-400">Cycle:</span>
                      {completedTestsCount <= 8 ? (
                        // Ha 8 vagy kevesebb teszt van, maradnak a gombok
                        <div className="flex gap-1">
                          {Array.from({ length: completedTestsCount }).map((_, i) => (
                            <button
                              key={i}
                              onClick={() => setSelectedCycle(i)}
                              className={`px-2 py-0.5 text-xs font-mono rounded cursor-pointer transition-all ${
                                selectedCycle === i 
                                  ? 'bg-emerald-600 text-white font-bold' 
                                  : 'bg-slate-800 text-slate-400 hover:text-white'
                              }`}
                            >
                              #{i}
                            </button>
                          ))}
                        </div>
                      ) : (
                        // Ha több mint 8 teszt van, egy tiszta legördülő menüt adunk neki
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

                {/* DEDIKÁLT KÉPMÉRET */}
                <div className="flex-1 flex items-center justify-center relative group border border-slate-800/40 rounded-xl overflow-hidden bg-white p-4 shadow-inner min-h-[300px] max-h-[520px]">
                  <div className="absolute top-4 right-4 bg-slate-900/90 text-white px-3 py-1.5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none flex items-center gap-1 text-xs font-medium z-10 shadow-md">
                    <ZoomIn className="w-4 h-4 text-emerald-400" /> Click to view full screen
                  </div>
                  <img
                    src={activeImageUrl}
                    alt="MPDRNN Phase Confusion Matrix"
                    onClick={() => setIsZoomed(true)}
                    className="w-full h-full object-contain cursor-zoom-in transition-transform duration-300 group-hover:scale-[1.01]"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = '';
                    }}
                  />
                </div>

                {/* Excel elérési út */}
                <p className="text-[11px] text-slate-400 text-center shrink-0">
                  Excel Saved to: <code className="bg-slate-950 px-2 py-1 rounded text-emerald-400 font-mono">{results.output_file}</code>
                </p>
              </div>
            ) : (
              /* Alapértelmezett üres állapot */
              <div className="text-center space-y-2 py-12">
                <ImageIcon className="w-12 h-12 text-slate-500 mx-auto" />
                <h4 className="font-semibold text-sm">No Active Plot Data</h4>
                <p className="text-xs text-slate-500 max-w-xs">Run the MPDRNN network to generate and display the training phase-plots.</p>
              </div>
            )}
          </div>

        </div>

      </div>

      {/* NAGYÍTOTT MODAL OVERLAY */}
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
              alt="MPDRNN Fullscreen Confusion Matrix"
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