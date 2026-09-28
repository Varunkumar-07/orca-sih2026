import { lazy } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Layout } from './components/Layout'
import { HomePage } from './pages/HomePage'

// Code-split every page but Home (the landing route, kept eager so the
// first paint never waits on a chunk fetch) — each of these, plus the
// leaflet/recharts dependencies only they use, now loads on first visit
// instead of bloating the initial bundle every visitor pays for up front.
const ChatMapPage = lazy(() => import('./pages/ChatMapPage'))
const ZonesExplorerPage = lazy(() => import('./pages/ZonesExplorerPage').then((m) => ({ default: m.ZonesExplorerPage })))
const WeatherPage = lazy(() => import('./pages/WeatherPage').then((m) => ({ default: m.WeatherPage })))
const RoutePlannerPage = lazy(() => import('./pages/RoutePlannerPage').then((m) => ({ default: m.RoutePlannerPage })))
const AlertsPage = lazy(() => import('./pages/AlertsPage').then((m) => ({ default: m.AlertsPage })))
const AnalyticsPage = lazy(() => import('./pages/AnalyticsPage').then((m) => ({ default: m.AnalyticsPage })))
const DownloadPage = lazy(() => import('./pages/DownloadPage').then((m) => ({ default: m.DownloadPage })))
const HistoryPage = lazy(() => import('./pages/HistoryPage').then((m) => ({ default: m.HistoryPage })))

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<HomePage />} />
          <Route path="/chat" element={<ChatMapPage />} />
          <Route path="/zones" element={<ZonesExplorerPage />} />
          <Route path="/weather" element={<WeatherPage />} />
          <Route path="/route" element={<RoutePlannerPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
          <Route path="/download" element={<DownloadPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="*" element={<HomePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
