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
    test_percentage: string;
    num_train_samples: number;
    num_test_samples: number;
  };
  saved_output_path: string;
}

export default function DatasetWorkspace({ darkMode, onBack }: DatasetWorkspaceProps) {
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [datasetName, setDatasetName] = useState<string>('');
  const [trainRatio, setTrainRatio] = useState<number>(0.8);

  // Állapotkezelés
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string>('');
  const [results, setResults] = useState<SplitResults | null>(null);

  // Dataset lista betöltése a közös :8000-es portról
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

  const handleConvertAndSplit = async () => {
    setStatus('loading');
    setErrorMessage('');
    setResults(null);

    try {
      const queryParams = `dataset=${datasetName}&train_ratio=${trainRatio}`;
      const response = await fetch(`http://localhost:8000/dataset/convert?${queryParams}`, {
        method: 'POST'
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

      {/* FŐ RÁCS (Kétoszlopos elrendezés) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-stretch">

        {/* BAL OSZLOP: PARAMÉTEREK ÉS NAVIGÁCIÓ */}
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

            {/* Vizuális Csúszka Csoport */}
            <div className="space-y-2 pt-2">
              <div className="flex justify-between text-xs font-semibold">
                <span className="text-blue-500">Train Ratio: {Math.round(trainRatio * 100)}%</span>
                <span className={darkMode ? 'text-slate-400' : 'text-slate-600'}>Test Ratio: {Math.round((1 - trainRatio) * 100)}%</span>
              </div>
              <input
                type="range"
                min="0.1"
                max="0.9"
                step="0.05"
                value={trainRatio}
                disabled={status === 'loading'}
                onChange={(e) => setTrainRatio(parseFloat(e.target.value))}
                className="w-full h-2 bg-slate-200 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-blue-600 disabled:opacity-50"
              />
            </div>

            {/* ÉLŐ VIZUÁLIS ELOSZLÁS JELZŐ */}
            <div className="space-y-1.5 pt-2">
              <span className={`block text-[11px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                Live Split Distribution
              </span>
              <div className="w-full h-6 rounded-lg overflow-hidden flex font-mono text-[10px] font-bold text-white text-center items-center shadow-inner">
                <div
                  className="bg-blue-600 transition-all duration-300 ease-out h-full flex items-center justify-center min-w-[40px]"
                  style={{ width: `${trainRatio * 100}%` }}
                >
                  {Math.round(trainRatio * 100)}%
                </div>
                <div
                  className="bg-slate-700 transition-all duration-300 ease-out h-full flex items-center justify-center flex-1"
                >
                  {Math.round((1 - trainRatio) * 100)}%
                </div>
              </div>
            </div>
          </div>

          {/* GOMBOK: INDÍTÁS ÉS VISSZA A DASHBOARDRA */}
          <div className="pt-4 space-y-2">
            <button
              onClick={handleConvertAndSplit}
              disabled={status === 'loading'}
              className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:bg-blue-800/40 text-white font-bold rounded-xl text-xs transition-all flex items-center justify-center gap-2 cursor-pointer shadow-md shadow-blue-600/10"
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
                  <h4 className={`font-bold text-sm ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>Executing Split Pipeline</h4>
                  <p className={`text-xs max-w-sm mx-auto ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                    Reading lines, mapping raw labels, shuffling with seed, applying MinMaxScaler and generating compiled cache matrices.
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

                {/* KPI KÁRTYÁK AZ EREDMÉNYEKKEL */}
                <div className="grid grid-cols-2 gap-4">
                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Training Samples
                    </span>
                    <span className="text-2xl font-mono font-bold text-blue-500">
                      {results.dynamic_split_results.num_train_samples.toLocaleString()}
                    </span>
                    <span className={`block text-[10px] font-mono mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                      Ratio: {results.dynamic_split_results.train_percentage}
                    </span>
                  </div>

                  <div className={`p-4 rounded-xl border ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                    <span className={`block text-[10px] font-medium uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                      Testing Samples
                    </span>
                    <span className="text-2xl font-mono font-bold text-amber-500">
                      {results.dynamic_split_results.num_test_samples.toLocaleString()}
                    </span>
                    <span className={`block text-[10px] font-mono mt-1 ${darkMode ? 'text-slate-500' : 'text-slate-600'}`}>
                      Ratio: {results.dynamic_split_results.test_percentage}
                    </span>
                  </div>
                </div>

                {/* JAVÍTVA: Total Processed Samples kontrasztja Light mode-ban */}
                <div className={`p-4 rounded-xl border flex justify-between items-center ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <span className={`text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    Total Processed Samples
                  </span>
                  <span className={`text-md font-mono font-bold ${darkMode ? 'text-slate-200' : 'text-slate-800'}`}>
                    {(results.dynamic_split_results.num_train_samples + results.dynamic_split_results.num_test_samples).toLocaleString()}
                  </span>
                </div>

                {/* JAVÍTVA: Elmentett fájl elérési út kódblokkja Light mode-ban */}
                <div className={`p-4 rounded-xl border space-y-2 ${darkMode ? 'bg-slate-950/40 border-slate-800' : 'bg-slate-50 border-slate-200'}`}>
                  <div className={`flex items-center gap-2 text-xs font-bold uppercase tracking-wider ${darkMode ? 'text-slate-400' : 'text-slate-600'}`}>
                    <FileCode className="w-4 h-4 text-blue-500" />
                    <span>Compiled Cache Artifact Path</span>
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
                  Select a target dataset and adjust the ratio slider to re-compile and split data arrays for network services.
                </p>
              </div>
            )}

          </div>
        </div>

      </div>
    </div>
  );
}