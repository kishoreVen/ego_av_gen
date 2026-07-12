import { Routes, Route } from 'react-router-dom';
import { RunsList } from './pages/RunsList';
import { RunDetail } from './pages/RunDetail';

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <a href="/" className="app-title">Brain Factory Watch</a>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/" element={<RunsList />} />
          <Route path="/run/:experiment/:timestamp" element={<RunDetail />} />
        </Routes>
      </main>
    </div>
  );
}
