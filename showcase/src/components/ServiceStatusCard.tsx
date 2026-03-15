interface ServiceStatusCardProps {
  name: string
  status: 'running' | 'deployed' | 'idle' | 'warning'
  description?: string
  icon?: string
}

const statusConfig = {
  running: {
    color: 'bg-green-500/20 border-green-500/50 text-green-600 dark:text-green-400',
    badge: '🟢',
    label: 'Running',
  },
  deployed: {
    color: 'bg-blue-500/20 border-blue-500/50 text-blue-600 dark:text-blue-400',
    badge: '🔵',
    label: 'Deployed',
  },
  idle: {
    color: 'bg-gray-500/20 border-gray-500/50 text-gray-600 dark:text-gray-400',
    badge: '⚪',
    label: 'Idle',
  },
  warning: {
    color: 'bg-amber-500/20 border-amber-500/50 text-amber-600 dark:text-amber-400',
    badge: '🟡',
    label: 'Warning',
  },
}

export default function ServiceStatusCard({
  name,
  status,
  description,
  icon,
}: ServiceStatusCardProps) {
  const config = statusConfig[status]

  return (
    <div className="card p-6 group hover:shadow-2xl">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-start gap-3 flex-1">
          <span className="text-2xl mt-0.5">{icon || '⚙️'}</span>
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-white text-lg">
              {name}
            </h3>
            {description && (
              <p className="text-sm text-zinc-600 dark:text-zinc-400 mt-1">
                {description}
              </p>
            )}
          </div>
        </div>
      </div>

      <div className={`inline-flex items-center gap-2 px-3 py-1 rounded-full border ${config.color}`}>
        <span className="text-sm">{config.badge}</span>
        <span className="text-xs font-semibold">{config.label}</span>
      </div>

      <div className="mt-4 h-1 bg-gradient-to-r from-accent/50 to-transparent rounded-full" />
    </div>
  )
}
