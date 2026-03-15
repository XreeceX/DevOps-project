import { useState, useEffect } from 'react'

export default function Navbar() {
  const [isDark, setIsDark] = useState(false)

  useEffect(() => {
    const theme = localStorage.getItem('theme')
    const isDarkMode = theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)
    setIsDark(isDarkMode)
    updateTheme(isDarkMode)
  }, [])

  const updateTheme = (dark: boolean) => {
    if (dark) {
      document.documentElement.classList.add('dark')
      localStorage.setItem('theme', 'dark')
    } else {
      document.documentElement.classList.remove('dark')
      localStorage.setItem('theme', 'light')
    }
  }

  const toggleTheme = () => {
    const newIsDark = !isDark
    setIsDark(newIsDark)
    updateTheme(newIsDark)
  }

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id)
    element?.scrollIntoView({ behavior: 'smooth' })
  }

  const navLinks = [
    { label: 'Overview', id: 'overview' },
    { label: 'Metrics', id: 'metrics' },
    { label: 'Visualizations', id: 'visualizations' },
    { label: 'Tech Stack', id: 'tech-stack' },
  ]

  return (
    <header className="sticky top-0 z-50 border-b border-zinc-200 dark:border-white/10 bg-white/80 dark:bg-[#0a0a0f]/80 backdrop-blur-lg">
      <div className="mx-auto max-w-6xl px-6 flex h-16 items-center justify-between">
        <div className="flex items-center gap-2">
          <a href="#" className="text-xl font-bold text-zinc-900 dark:text-white hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
            RR
          </a>
          <span className="text-zinc-400 dark:text-zinc-700">—</span>
          <h1 className="text-sm font-semibold text-zinc-600 dark:text-zinc-400">
            Fraud Detection
          </h1>
        </div>

        <nav className="hidden md:flex items-center gap-8">
          {navLinks.map((link) => (
            <button
              key={link.id}
              onClick={() => scrollToSection(link.id)}
              className="text-sm font-medium text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors"
            >
              {link.label}
            </button>
          ))}
        </nav>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
          aria-label="Toggle theme"
        >
          {isDark ? (
            <span className="text-lg">☀️</span>
          ) : (
            <span className="text-lg">🌙</span>
          )}
        </button>
      </div>
    </header>
  )
}
