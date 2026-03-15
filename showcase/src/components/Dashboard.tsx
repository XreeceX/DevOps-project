import ServiceStatusCard from './ServiceStatusCard'

const SERVICES = [
  {
    name: 'Frontend',
    status: 'deployed' as const,
    description: 'React + Vite application',
    icon: '⚛️',
  },
  {
    name: 'API Server',
    status: 'running' as const,
    description: 'Backend services',
    icon: '🔗',
  },
  {
    name: 'Database',
    status: 'running' as const,
    description: 'PostgreSQL',
    icon: '🗄️',
  },
  {
    name: 'Cache',
    status: 'running' as const,
    description: 'Redis cache layer',
    icon: '⚡',
  },
]

const DEPLOYMENT_INFO = [
  { label: 'Platform', value: 'Vercel' },
  { label: 'Framework', value: 'React + Vite' },
  { label: 'Region', value: 'Edge Network' },
  { label: 'Branch', value: 'main' },
]

const ENVIRONMENT_DETAILS = [
  { label: 'Node Version', value: '18+' },
  { label: 'Package Manager', value: 'npm' },
  { label: 'Language', value: 'TypeScript' },
  { label: 'Styling', value: 'TailwindCSS' },
]

export default function Dashboard() {
  return (
    <main className="min-h-screen bg-gradient-to-b from-white to-zinc-50 dark:from-[#0a0a0f] dark:to-zinc-900/50">
      <div className="mx-auto max-w-6xl px-6 py-16">
        {/* Hero Section */}
        <section className="mb-24 relative">
          <div className="absolute -inset-x-20 -top-20 h-96 bg-gradient-to-b from-accent/10 to-transparent rounded-full blur-3xl -z-10 dark:from-accent/5" />

          <div className="space-y-6">
            <div className="inline-block">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-accent/10 border border-accent/30">
                <span className="w-2 h-2 bg-accent rounded-full animate-pulse" />
                <p className="text-sm font-semibold text-accent">System Status</p>
              </div>
            </div>

            <h1 className="text-5xl lg:text-6xl font-bold tracking-tight text-zinc-900 dark:text-white leading-tight">
              DevOps<br />Dashboard
            </h1>

            <p className="text-lg text-zinc-700 dark:text-zinc-400 max-w-2xl">
              Monitor your deployment infrastructure, service status, and application health in real-time.
            </p>
          </div>
        </section>

        {/* Services Status Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Services Status</h2>
            <p className="section-subtitle">Real-time monitoring of all applications and services</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {SERVICES.map((service) => (
              <ServiceStatusCard
                key={service.name}
                name={service.name}
                status={service.status}
                description={service.description}
                icon={service.icon}
              />
            ))}
          </div>
        </section>

        {/* Deployment Info Section */}
        <section className="mb-24">
          <div className="mb-12">
            <h2 className="section-title">Deployment Information</h2>
            <p className="section-subtitle">Current deployment configuration and details</p>
          </div>

          <div className="card p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {DEPLOYMENT_INFO.map((item) => (
                <div key={item.label} className="space-y-2">
                  <p className="text-sm uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400">
                    {item.label}
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-white">
                    {item.value}
                  </p>
                  <div className="h-1 w-8 bg-accent rounded-full" />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Environment Details Section */}
        <section className="mb-16">
          <div className="mb-12">
            <h2 className="section-title">Environment Details</h2>
            <p className="section-subtitle">Technologies and runtime configuration</p>
          </div>

          <div className="card p-8">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
              {ENVIRONMENT_DETAILS.map((item) => (
                <div key={item.label} className="space-y-2">
                  <p className="text-sm uppercase tracking-widest font-semibold text-zinc-600 dark:text-zinc-400">
                    {item.label}
                  </p>
                  <p className="text-lg font-bold text-zinc-900 dark:text-white">
                    {item.value}
                  </p>
                  <div className="h-1 w-8 bg-accent rounded-full" />
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Footer */}
        <footer className="border-t border-zinc-200 dark:border-white/10 pt-12 pb-6">
          <p className="text-sm text-zinc-700 dark:text-zinc-600">
            © 2026 DevOps Project. Built with React + Vite + TailwindCSS
          </p>
        </footer>
      </div>
    </main>
  )
}
