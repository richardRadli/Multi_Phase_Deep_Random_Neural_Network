import { useState, useEffect } from 'react';
import {
  Database,
  Cpu,
  Network,
  Binary,
  Moon,
  Sun
} from 'lucide-react';

import MpdrnnWorkspace from './views/MpdrnnWorkspace';
import HelmWorkSpace from './views/HelmWorkspace';
import DatasetWorkspace from './views/DatasetWorkspace';
import FcnnWorkspace from './views/FcnnWorkspace';

// 💡 1. KIEGÉSZÍTVE A TUNE MÓDDAL
type ViewType = 'dashboard' | 'dataset' | 'fcnn_train' | 'fcnn_test' | 'fcnn_tune' | 'helm' | 'helm_tune' | 'mpdrnn';

interface ServiceStatus {
  dataset: 'online' | 'offline' | 'checking';
  fcnn: 'online' | 'offline' | 'checking';
  helm: 'online' | 'offline' | 'checking';
  mpdrnn: 'online' | 'offline' | 'checking';
}

export default function App() {
  const [darkMode, setDarkMode] = useState(false);
  const [currentView, setCurrentView] = useState<ViewType>('dashboard');

  const [services, setServices] = useState<ServiceStatus>({
    dataset: 'checking',
    fcnn: 'checking',
    helm: 'checking',
    mpdrnn: 'checking'
  });

  const checkEndpoint = async (url: string): Promise<boolean> => {
    try {
      const controller = new AbortController();
      const id = setTimeout(() => controller.abort(), 1500); // 1.5 mp timeout

      const res = await fetch(url, {
        signal: controller.signal,
        cache: 'no-cache'
      });

      clearTimeout(id);
      return res.ok;
    } catch {
      return false;
    }
  };

  // Státuszellenőrző ciklus
  useEffect(() => {
    const checkAllServices = async () => {
      const [datasetOk, fcnnOk, helmOk, mpdrnnOk] = await Promise.all([
        checkEndpoint('http://localhost:8000/'),
        checkEndpoint('http://localhost:8001/'),
        checkEndpoint('http://localhost:8002/'),
        checkEndpoint('http://localhost:8003/')
      ]);

      setServices({
        dataset: datasetOk ? 'online' : 'offline',
        fcnn: fcnnOk ? 'online' : 'offline',
        helm: helmOk ? 'online' : 'offline',
        mpdrnn: mpdrnnOk ? 'online' : 'offline'
      });
    };

    void checkAllServices();

    const interval = setInterval(() => {
      void checkAllServices();
    }, 8000); // 8 másodpercenként frissít

    return () => clearInterval(interval);
  }, []);

  // Státuszjelző pilula a fejlécben
  const StatusDot = ({ label, port, status }: { label: string; port: string; status: 'online' | 'offline' | 'checking' }) => {
    return (
      <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all duration-300 text-xs font-semibold ${
        darkMode 
          ? 'bg-slate-900 border-slate-800 text-slate-300 shadow-inner' 
          : 'bg-slate-100 border-slate-200 text-slate-600 shadow-sm'
      }`}>
        <span className="relative flex h-2.5 w-2.5">
          {status === 'online' && (
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
          )}
          <span className={`relative inline-flex rounded-full h-2.5 w-2.5 transition-all duration-300 ${
            status === 'online' ? 'bg-emerald-500' : status === 'checking' ? 'bg-amber-500 animate-pulse' : 'bg-rose-500'
          }`}></span>
        </span>
        <span className="font-mono tracking-tight text-[11px]">
          {label} <span className={darkMode ? 'text-slate-600' : 'text-slate-400'}>:{port}</span>
        </span>
      </div>
    );
  };

  return (
    <div className={`min-h-screen flex flex-col justify-between transition-colors duration-200 ${
      darkMode ? 'bg-slate-950 text-slate-100' : 'bg-slate-50 text-slate-900'
    }`}>

      {/* HEADER */}
      <header className={`border-b px-8 py-4 flex items-center justify-between shrink-0 transition-colors duration-200 ${
        darkMode ? 'border-slate-800 bg-slate-900' : 'border-slate-200 bg-white shadow-sm'
      }`}>
        {/* Bal oldal: Logó */}
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-blue-600 text-white rounded-xl shadow-md shadow-blue-500/20">
            <Cpu className="w-6 h-6 animate-pulse" />
          </div>
          <h1 className="text-xl font-extrabold tracking-tight">Neural Network Training and Eval System</h1>
        </div>

        {/* Jobb oldal: Élő Státuszok */}
        <div className="flex items-center gap-6">
          <div className="hidden md:flex items-center gap-2.5">
            <StatusDot label="DATA" port="8000" status={services.dataset} />
            <StatusDot label="FCNN" port="8001" status={services.fcnn} />
            <StatusDot label="HELM" port="8002" status={services.helm} />
            <StatusDot label="MPDR" port="8003" status={services.mpdrnn} />
          </div>

          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`p-2 rounded-xl border cursor-pointer transition-all duration-200 ${
              darkMode ? 'border-slate-700 hover:bg-slate-800 text-amber-400' : 'border-slate-200 hover:bg-slate-100 text-slate-700'
            }`}
          >
            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
          </button>
        </div>
      </header>

      {/* RENDERELT NÉZETEK */}
      <main className={`flex-1 w-full max-w-[1600px] mx-auto p-6 md:p-8 flex flex-col justify-center`}>

        {/* ==================== 1. DASHBOARD NÉZET ==================== */}
        {currentView === 'dashboard' && (
          <div className="space-y-8 w-full animate-fadeIn max-w-6xl mx-auto py-4">

            <div className="text-center space-y-2 mb-4">
              <h2 className="text-3xl font-black tracking-tight bg-gradient-to-r from-blue-500 via-indigo-500 to-emerald-500 bg-clip-text text-transparent">
                Main Control Panel
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full">

              {/* Dataset Card */}
              <div className={`p-8 rounded-2xl border transition-all duration-300 hover:scale-[1.015] hover:shadow-xl flex flex-col justify-between min-h-[220px] ${
                darkMode ? 'bg-slate-900 border-slate-800 hover:border-blue-500/30' : 'bg-white border-slate-200 hover:shadow-blue-500/5 shadow-md'
              }`}>
                <div>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                      <div className="p-3.5 bg-blue-50 text-blue-600 dark:bg-blue-950/40 dark:text-blue-400 rounded-2xl shadow-sm">
                        <Database className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="font-extrabold text-lg">Dataset Operations</h3>
                        <span className="text-xs font-mono text-slate-500">Port :8000</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed mb-6">Split data, prepare, convert and manage datasets for training.</p>
                </div>
                <button
                  onClick={() => setCurrentView('dataset')}
                  className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white font-bold rounded-xl text-xs cursor-pointer transition-all shadow-md shadow-blue-600/10"
                >
                  Open Dataset Workspace
                </button>
              </div>

              {/* FCNN Card - 💡 2. HARMINCADIK GOMBBAL KIEGÉSZÍTVE */}
              <div className={`p-8 rounded-2xl border transition-all duration-300 hover:scale-[1.015] hover:shadow-xl flex flex-col justify-between min-h-[220px] ${
                darkMode ? 'bg-slate-900 border-slate-800 hover:border-indigo-500/30' : 'bg-white border-slate-200 hover:shadow-indigo-500/5 shadow-md'
              }`}>
                <div>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                      <div className="p-3.5 bg-indigo-50 text-indigo-600 dark:bg-indigo-950/40 dark:text-indigo-400 rounded-2xl shadow-sm">
                        <Cpu className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="font-extrabold text-lg">FCNN Service</h3>
                        <span className="text-xs font-mono text-slate-500">Port :8001</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed mb-6">Fully Connected Neural Network training, evaluation and hyperparameter tuning workspace.</p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5">
                  <button
                    onClick={() => setCurrentView('fcnn_train')}
                    className="py-3 bg-indigo-600 hover:bg-indigo-500 text-white font-bold rounded-xl text-[11px] cursor-pointer transition-all shadow-md shadow-indigo-600/10 text-center"
                  >
                    Training
                  </button>
                  <button
                    onClick={() => setCurrentView('fcnn_test')}
                    className="py-3 font-bold rounded-xl text-[11px] cursor-pointer transition-all bg-slate-800 hover:bg-slate-700 text-white border border-slate-700 shadow-md text-center"
                  >
                    Testing
                  </button>
                  <button
                    onClick={() => setCurrentView('fcnn_tune')}
                    className="py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl text-[11px] cursor-pointer transition-all shadow-md shadow-purple-600/10 text-center"
                  >
                    H-Param Tuning
                  </button>
                </div>
              </div>

              {/* HELM Card */}
              <div className={`p-8 rounded-2xl border transition-all duration-300 hover:scale-[1.015] hover:shadow-xl flex flex-col justify-between min-h-[220px] ${
                darkMode ? 'bg-slate-900 border-slate-800 hover:border-violet-500/30' : 'bg-white border-slate-200 hover:shadow-violet-500/5 shadow-md'
              }`}>
                <div>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                      <div className="p-3.5 bg-violet-50 text-violet-600 dark:bg-violet-950/40 dark:text-violet-400 rounded-2xl shadow-sm">
                        <Network className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="font-extrabold text-lg">HELM Service</h3>
                        <span className="text-xs font-mono text-slate-500">Port :8002</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed mb-6">Hierarchical Extreme Learning Machine service.</p>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                  <button
                    onClick={() => setCurrentView('helm')}
                    className="py-3 bg-violet-600 hover:bg-violet-500 text-white font-bold rounded-xl text-[11px] cursor-pointer transition-all shadow-md shadow-violet-600/10 text-center"
                  >
                    Execution
                  </button>
                  <button
                    onClick={() => setCurrentView('helm_tune')}
                    className="py-3 bg-purple-600 hover:bg-purple-500 text-white font-bold rounded-xl text-[11px] cursor-pointer transition-all shadow-md shadow-purple-600/10 text-center"
                  >
                    H-Param Tuning
                  </button>
                </div>
              </div>

              {/* MPDRNN Card */}
              <div className={`p-8 rounded-2xl border transition-all duration-300 hover:scale-[1.015] hover:shadow-xl flex flex-col justify-between min-h-[220px] ${
                darkMode ? 'bg-slate-900 border-slate-800 hover:border-emerald-500/30' : 'bg-white border-slate-200 hover:shadow-emerald-500/5 shadow-md'
              }`}>
                <div>
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-4">
                      <div className="p-3.5 bg-emerald-50 text-emerald-600 dark:bg-emerald-950/40 dark:text-emerald-400 rounded-2xl shadow-sm">
                        <Binary className="w-7 h-7" />
                      </div>
                      <div>
                        <h3 className="font-extrabold text-lg">MPDRNN Service</h3>
                        <span className="text-xs font-mono text-slate-500">Port :8003</span>
                      </div>
                    </div>
                  </div>
                  <p className="text-sm text-slate-500 leading-relaxed mb-6">Multi-Phase Deep Randomized Neural Network control center.</p>
                </div>
                <button
                  onClick={() => setCurrentView('mpdrnn')}
                  className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 text-white font-bold rounded-xl text-xs cursor-pointer transition-all shadow-md shadow-emerald-600/10"
                >
                  Open MPDRNN Workspace
                </button>
              </div>

            </div>
          </div>
        )}

        {/* WORKSPACE-EK */}
        {currentView !== 'dashboard' && (
          <div className="w-full h-full animate-fadeIn">
            {currentView === 'dataset' && (
              <DatasetWorkspace
                darkMode={darkMode}
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {currentView === 'fcnn_train' && (
              <FcnnWorkspace
                darkMode={darkMode}
                mode="train"
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {currentView === 'fcnn_test' && (
              <FcnnWorkspace
                darkMode={darkMode}
                mode="test"
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {/* 💡 3. TUNE MÓDÚ FCNN WORKSPACE RENDERELÉSE */}
            {currentView === 'fcnn_tune' && (
              <FcnnWorkspace
                darkMode={darkMode}
                mode="tune"
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {currentView === 'helm' && (
              <HelmWorkSpace
                darkMode={darkMode}
                mode="run"
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {currentView === 'helm_tune' && (
              <HelmWorkSpace
                darkMode={darkMode}
                mode="tune"
                onBack={() => setCurrentView('dashboard')}
              />
            )}

            {currentView === 'mpdrnn' && (
              <MpdrnnWorkspace
                darkMode={darkMode}
                onBack={() => setCurrentView('dashboard')}
              />
            )}
          </div>
        )}

      </main>

      {/* FOOTER */}
      <footer className={`w-full py-4 text-center shrink-0 border-t transition-colors duration-200 ${
        darkMode ? 'border-slate-800/80 bg-slate-950/20 text-slate-500' : 'border-slate-200 bg-slate-100/50 text-slate-400'
      }`}>
        <p className="text-xs font-semibold tracking-wide">© 2026 University of Pannonia - Image Processing Laboratory. All rights reserved.</p>
      </footer>
    </div>
  );
}