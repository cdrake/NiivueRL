import { HashRouter, Route, Routes } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import MainApp from './pages/MainApp';
import RunExperimentPage from './pages/RunExperimentPage';
import NotesPage from './pages/NotesPage';
import './App.css';

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/run/:slug" element={<RunExperimentPage />} />
        <Route path="/configure" element={<MainApp initialMode="experiment" />} />
        <Route path="/interactive" element={<MainApp initialMode="interactive" />} />
        <Route path="/notes" element={<NotesPage />} />
        <Route path="*" element={<LandingPage />} />
      </Routes>
    </HashRouter>
  );
}
