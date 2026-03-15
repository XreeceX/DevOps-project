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

  return (
    <header className="sticky top-0 z-50 border-b border-zinc-200 dark:border-white/10 bg-white/80 dark:bg-[#0a0a0f]/80 backdrop-blur-lg">
      <div className="mx-auto max-w-6xl px-6 flex h-16 items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent to-amber-400 flex items-center justify-center">
            <span className="text-sm font-bold text-black">⚙️</span>
          </div>
          <h1 className="text-lg font-bold text-zinc-900 dark:text-white">
            DevOps Dashboard
          </h1>
        </div>

        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg hover:bg-zinc-200 dark:hover:bg-white/10 transition-colors"
          aria-label="Toggle theme"
        >
          {isDark ? (
            <span className="text-xl">☀️</span>
          ) : (
            <span className="text-xl">🌙</span>
          )}
        </button>
      </div>
    </header>
  )
}
