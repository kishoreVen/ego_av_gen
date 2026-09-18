import { Routes, Route, Link, useLocation } from 'react-router-dom';
import { RunsList } from './pages/RunsList';
import { RunDetail } from './pages/RunDetail';
import { SimViewer } from './pages/SimViewer';

export default function App() {
  const { pathname } = useLocation();
  return (
    <div className="app">
      <header className="app-header">
        <Link to="/" className="app-title">Brain Factory Watch</Link>
        <nav className="app-nav">
          <Link to="/"    className={`app-nav-link ${pathname === '/'    ? 'active' : ''}`}>Runs</Link>
          <Link to="/sim" className={`app-nav-link ${pathname === '/sim' ? 'active' : ''}`}>Robot Sim</Link>
        </nav>
      </header>
      <main className="app-main">
        <Routes>
          <Route path="/"                              element={<RunsList />} />
          <Route path="/run/:experiment/:timestamp"   element={<RunDetail />} />
          <Route path="/sim"                           element={<SimViewer />} />
        </Routes>
      </main>
    </div>
  );
}
