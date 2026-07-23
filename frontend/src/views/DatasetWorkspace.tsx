import { useState, useEffect } from 'react';
import { ArrowLeft, Database, Sliders, RefreshCw, CheckCircle, AlertCircle, FileCode } from 'lucide-react';

interface DatasetWorkspaceProps {
  darkMode: boolean;
  onBack: () => void;
}

interface SplitResults {
  status: string;
  message: string;
  dynamic_split_results: {
    train_percentage: string;
    valid_percentage: string;
    test_percentage: string;
    num_train_samples: number;
    num_valid_samples: number;
    num_test_samples: number;
    total_samples: number;
  };
  saved_output_path: string;
}

export default function DatasetWorkspace({ darkMode, onBack }: DatasetWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');

  // Százalékos értékek 0-100 közötti egész számokként
  const [trainPct, setTrainPct] = useState<number>(70);
  const [validPct, setValidPct] = useState<number>(20);
  const [testPct, setTestPct] = useState<number>(10);

  // Állapotkezelés
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [results, setResults] = useState<SplitResults | null>(null);

  const totalPct = trainPct + validPct + testPct;

  // Dataset lista betöltése
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
        const fallback = ['connect4', 'mnist', 'letter'];
        setAvailableDatasets(fallback);
        setDatasetName(fallback[0]);
      }
    };

    void fetchDatasets();
  }, []);

  // Presetek gyors beállítása
  const applyPreset = (train: number, valid: number, test: number) => {
    setTrainPct(train);
    setValidPct(valid);
    setTestPct(test);
  };

  // Változáskezelők automatikus korrekcióval
  const handleTrainChange = (val: number) => {
    const newTrain = Math.min(100, Math.max(0, val));
    setTrainPct(newTrain);
    const remaining = 100 - newTrain;
    if (validPct + testPct === 0) {
      setValidPct(Math.round(remaining * 0.5));
      setTestPct(Math.round(remaining * 0.5));
    } else {
      const validShare = validPct / (validPct + testPct);
      const newValid = Math.round(remaining * validShare);
      setValidPct(newValid);
      setTestPct(100 - newTrain - newValid);
    }
  };

  const handleValidChange = (val: number) => {
    const newValid = Math.min(100 - trainPct, Math.max(0, val));
    setValidPct(newValid);
    setTestPct(100 - trainPct - newValid);
  };

  const handleTestChange = (val: number) => {
    const newTest = Math.min(100 - trainPct, Math.max(0, val));
    setTestPct(newTest);
    setValidPct(100 - trainPct - newTest);
  };

  const handleConvertAndSplit = async () => {
    if (totalPct !== 100) {
      setErrorMessage('A három érték összegének pontosan 100%-nak kell lennie!');
      setStatus('error');
      return;
    }

    setStatus('loading');
    setErrorMessage('');
    setResults(null);

    try {
      const response = await fetch('http://localhost:8000/dataset/convert', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset: datasetName,
          split_ratio: [
            Number((trainPct / 100).toFixed(2)),
            Number((validPct / 100).toFixed(2)),
            Number((testPct / 100).toFixed(2))
          ]
        })
      });

      if (!response.ok) {
        const errData = await response.json() as { detail?: string };
        throw new Error(errData.detail || 'Failed to split and convert dataset.');
      }

      const data = (await response.json()) as SplitResults;
      setResults(data);
      setStatus('success');
    } catch (err) {
      setStatus('error');
      setErrorMessage(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className="space-y-6">
      {/* CÍMSOR ÉS MÓD JELZŐ */}
      <div className="flex items-center justify-end">
        <span className="text-xs font-mono text-blue-500">Dataset Operations Lab</span>
      </div>

      {/* FŐ RÁCS */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* BAL OSZLOP: PARAMÉTEREK ÉS BEVITELI MEZŐK */}
        <div className={`p-6 rounded-2xl border ${
          darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
        } flex flex-col justify-between h-full shadow-md`}>
          <div className="space-y-5">
            <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
              <Sliders className="w-5 h-5 text-blue-500" />
              <h3 className={`text-lg font-semibold ${darkMode ? 'text-slate-100' : 'text-slate-800'}`}>Configuration</h3>
            </div>

            {/* Dataset Választó */}
            <div>
              <label className={`block text-xs font-medium mb-1 ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>Target Dataset</label>
              <select
                value={datasetName}
                disabled={status === 'loading'}
                onChange={(e) => setDatasetName(e.target.value)}
                className={`w-full text-sm p-2.5 rounded-md border ${
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

            {/* PRESETEK */}
            <div className="space-y-1.5">
              <label className={`block text-[11px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Quick Split Presets
              </label>
              <div className="grid grid-cols-3 gap-2">
                <button
                  type="button"
                  onClick={() => applyPreset(70, 15, 15)}
                  disabled={status === 'loading'}
                  className={`py-1.5 text-xs font-mono font-semibold rounded-lg border transition-all ${
                    trainPct === 70 && validPct === 15 && testPct === 15
                      ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                      : darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                      : 'bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  70 / 15 / 15
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset(70, 20, 10)}
                  disabled={status === 'loading'}
                  className={`py-1.5 text-xs font-mono font-semibold rounded-lg border transition-all ${
                    trainPct === 70 && validPct === 20 && testPct === 10
                      ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                      : darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                      : 'bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  70 / 20 / 10
                </button>
                <button
                  type="button"
                  onClick={() => applyPreset(80, 10, 10)}
                  disabled={status === 'loading'}
                  className={`py-1.5 text-xs font-mono font-semibold rounded-lg border transition-all ${
                    trainPct === 80 && validPct === 10 && testPct === 10
                      ? 'bg-blue-600 text-white border-blue-500 shadow-sm'
                      : darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-300 hover:border-slate-700'
                      : 'bg-slate-100 border-slate-200 text-slate-700 hover:bg-slate-200'
                  }`}
                >
                  80 / 10 / 10
                </button>
              </div>
            </div>

            {/* 3 BEVITELI MEZŐ ÉS VIZUÁLIS ELEMEI (KÉP ALAPJÁN) */}
            <div className="space-y-4 pt-2">

              {/* TRAIN MEZŐ */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Train</span>
                  <span className="font-mono font-bold text-blue-500">{trainPct}%</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={trainPct}
                  disabled={status === 'loading'}
                  onChange={(e) => handleTrainChange(parseInt(e.target.value) || 0)}
                  className={`w-full text-sm font-semibold p-2.5 rounded-xl border transition-all ${
                    darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-200 focus:border-blue-500'
                      : 'bg-amber-50/40 border-stone-200 text-slate-800 focus:border-blue-500'
                  } focus:outline-none`}
                />
              </div>

              {/* VALIDATION MEZŐ */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Validation</span>
                  <span className="font-mono font-bold text-purple-500">{validPct}%</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={validPct}
                  disabled={status === 'loading'}
                  onChange={(e) => handleValidChange(parseInt(e.target.value) || 0)}
                  className={`w-full text-sm font-semibold p-2.5 rounded-xl border transition-all ${
                    darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-200 focus:border-purple-500'
                      : 'bg-amber-50/40 border-stone-200 text-slate-800 focus:border-purple-500'
                  } focus:outline-none`}
                />
              </div>

              {/* TEST MEZŐ */}
              <div className="space-y-1">
                <div className="flex justify-between items-center text-xs font-semibold">
                  <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Test</span>
                  <span className="font-mono font-bold text-amber-600">{testPct}%</span>
                </div>
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={testPct}
                  disabled={status === 'loading'}
                  onChange={(e) => handleTestChange(parseInt(e.target.value) || 0)}
                  className={`w-full text-sm font-semibold p-2.5 rounded-xl border transition-all ${
                    darkMode
                      ? 'bg-slate-950 border-slate-800 text-slate-200 focus:border-amber-500'
                      : 'bg-amber-50/40 border-stone-200 text-slate-800 focus:border-amber-500'
                  } focus:outline-none`}
                />
              </div>

              {/* A KÉPEN LÁTHATÓ ELOSZLÁS SÁV ÉS TOTAL JELZÉS */}
              <div className="space-y-2 pt-2">
                <div className="w-full h-3 rounded-full overflow-hidden flex shadow-inner bg-slate-200 dark:bg-slate-800">
                  <div
                    className="bg-blue-500 transition-all duration-300 h-full"
                    style={{ width: `${trainPct}%` }}
                  />
                  <div
                    className="bg-purple-500 transition-all duration-300 h-full"
                    style={{ width: `${validPct}%` }}
                  />
                  <div
                    className="bg-amber-500 transition-all duration-300 h-full"
                    style={{ width: `${testPct}%` }}
                  />
                </div>

                <div className="flex justify-end items-center text-xs font-bold font-mono">
                  {totalPct === 100 ? (
                    <span className="text-emerald-500 flex items-center gap-1">
                      Total: 100% ✓
                    </span>
                  ) : (
                    <span className="text-rose-500">
                      Total: {totalPct}% (Nincs 100%)
                    </span>
                  )}
                </div>
              </div>

            </div>
          </div>

          {/* GOMBOK */}
          <div className="pt-4 space-y-2">
            <button
              onClick={handleConvertAndSplit}
              disabled={status === 'loading' || totalPct !== 100}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-700 text-white font-bold rounded-xl text-xs transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md"
            >
              {status === 'loading' ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" /> Processing & Normalizing...
                </>
              ) : (
                <>
                  <Database className="w-4 h-4" /> Run Split & Preprocessing
                </>
              )}
            </button>

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

        {/* JOBB OSZLOP: LIVE ANALYTICS ÉS EREDMÉNYEK */}
        <div className="lg:col-span-2 flex flex-col h-full space-y-4">

          {/* Hibaüzenet */}
          {status === 'error' && (
            <div className={`p-4 rounded-xl border flex items-start gap-3 shrink-0 ${
              darkMode ? 'bg-rose-950/30 border-rose-900/50 text-rose-400' : 'bg-rose-50 border-rose-200 text-rose-700'
            }`}>
              <AlertCircle className="w-5 h-5 text-rose-500 shrink-0 mt-0.5" />
              <div>
                <h4 className="font-semibold text-sm">Operation Failed</h4>
                <p className="text-xs mt-1">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* Fő kijelző panel */}
          <div className={`flex-1 p-6 rounded-2xl border flex flex-col justify-center ${
            darkMode ? 'bg-slate-900 border-slate-800' : 'bg-white border-slate-200'
          } shadow-md`}>

            {status === 'loading' && (
              <div className="text-center space-y-4 py-12 animate-pulse">
                <RefreshCw className="w-12 h-12 text-blue-500 animate-spin mx-auto" />
                <div className="space-y-1">
                  <h4 className={`font-bold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>Executing 3-Way Split Pipeline</h4>
                  <p className={`text-xs max-w-sm mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                    Splitting dataset into Train, Validation, and Test sets, normalizing features, and compiling cache artifact.
                  </p>
                </div>
              </div>
            )}

            {status === 'success' && results && (
              <div className="space-y-6 animate-fadeIn w-full h-full flex flex-col justify-between">

                {/* Sikeres fejléc */}
                <div className={`flex items-center gap-3 border-b pb-3 ${darkMode ? 'border-slate-800/40' : 'border-slate-200'}`}>
                  <CheckCircle className="w-5 h-5 text-emerald-500" />
                  <h4 className="font-bold text-sm text-emerald-500">Pipeline Executed Successfully</h4>
                </div>

                {/* KPI KÁRTYÁK AZ EREDMÉNYEKKEL (3 HALMAZ) */}
                <div className="grid grid-cols-3 gap-3">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Training
                    </span>
                    <span className="text-xl font-mono font-bold text-blue-500">
                      {results.dynamic_split_results.num_train_samples.toLocaleString()}
                    </span>
                    <span className={`block text-[10px] font-mono mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                      Ratio: {results.dynamic_split_results.train_percentage}
                    </span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Validation
                    </span>
                    <span className="text-xl font-mono font-bold text-purple-500">
                      {results.dynamic_split_results.num_valid_samples.toLocaleString()}
                    </span>
                    <span className={`block text-[10px] font-mono mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                      Ratio: {results.dynamic_split_results.valid_percentage}
                    </span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Testing
                    </span>
                    <span className="text-xl font-mono font-bold text-amber-500">
                      {results.dynamic_split_results.num_test_samples.toLocaleString()}
                    </span>
                    <span className={`block text-[10px] font-mono mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                      Ratio: {results.dynamic_split_results.test_percentage}
                    </span>
                  </div>
                </div>

                {/* Total Processed Samples */}
                <div className={`p-4 rounded-xl border flex justify-between items-center ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    Total Processed Samples
                  </span>
                  <span className={`text-md font-mono font-bold ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                    {results.dynamic_split_results.total_samples.toLocaleString()}
                  </span>
                </div>

                {/* Elmentett fájl elérési út */}
                <div className={`p-4 rounded-xl border space-y-2 ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <div className={`flex items-center gap-2 text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    <FileCode className="w-4 h-4 text-blue-500" />
                    <span>Compiled Cache Artifact Path (.npz)</span>
                  </div>
                  <code className={`block p-2.5 rounded text-xs font-mono border overflow-x-auto break-all ${
                    darkMode
                      ? 'bg-slate-950 border-slate-800 text-blue-400'
                      : 'bg-slate-100 border-slate-200 text-blue-600'
                  }`}>
                    {results.saved_output_path}
                  </code>
                </div>
              </div>
            )}

            {status === 'idle' && (
              <div className="text-center space-y-2 py-12">
                <Database className={`w-12 h-12 mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-400'}`} />
                <h4 className={`font-semibold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>No Active Dataset Split</h4>
                <p className={`text-xs max-w-xs mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                  Select a target dataset and enter split percentages or use presets to compile data arrays.
                </p>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}