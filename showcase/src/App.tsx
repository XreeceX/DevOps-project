import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Dashboard from './components/Dashboard'
import Footer from './components/Footer'

function App() {
  return (
    <div className="min-h-screen bg-white dark:bg-[#0a0a0f]">
      <Navbar />
      <Hero />
      <Dashboard />
      <Footer />
    </div>
  )
}

export default App
