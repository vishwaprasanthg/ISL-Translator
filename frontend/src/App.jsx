import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './context/AppContext';
import Layout from './components/layout/Layout';
import Dashboard from './pages/Dashboard';
import Alphabet from './pages/Alphabet';
import Numbers from './pages/Numbers';
import Vocabulary from './pages/Vocabulary';
import History from './pages/History';
import Settings from './pages/Settings';
import About from './pages/About';

export default function App() {
  return (
    <AppProvider>
      <Router>
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/alphabet" element={<Alphabet />} />
            <Route path="/numbers" element={<Numbers />} />
            <Route path="/vocabulary" element={<Vocabulary />} />
            <Route path="/history" element={<History />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </Layout>
      </Router>
    </AppProvider>
  );
}
