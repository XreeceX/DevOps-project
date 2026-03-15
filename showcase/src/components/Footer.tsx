export default function Footer() {
  const currentYear = new Date().getFullYear()

  const links = [
    { label: 'GitHub', href: 'https://github.com/XreeceX/DevOps-project', icon: '→' },
    { label: 'Live Demo', href: 'https://dev-ops-project-six.vercel.app', icon: '→' },
    { label: 'Main Portfolio', href: 'https://reece-rodrigues.vercel.app', icon: '→' },
  ]

  return (
    <footer className="border-t border-zinc-200 dark:border-white/10 pt-16 pb-8">
      <div className="mx-auto max-w-6xl px-6">
        {/* CTA Section */}
        <div className="mb-16 rounded-2xl bg-gradient-to-br from-blue-50 dark:from-blue-950/40 to-purple-50 dark:to-purple-950/40 border border-blue-200/50 dark:border-blue-800/30 p-8 md:p-12">
          <div className="space-y-6 text-center">
            <h2 className="text-3xl md:text-4xl font-bold text-zinc-900 dark:text-white">
              Explore More Projects
            </h2>
            <p className="text-lg text-zinc-600 dark:text-zinc-400 max-w-2xl mx-auto">
              Check out my portfolio for more machine learning, AI, and full-stack projects.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <a
                href="https://reece-rodrigues.vercel.app"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary"
              >
                <span>View Full Portfolio</span>
                <span>→</span>
              </a>
            </div>
          </div>
        </div>

        {/* Links Section */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-12 mb-12">
          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-white mb-4">Project</h3>
            <ul className="space-y-2">
              {links.map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors flex items-center gap-2 group"
                  >
                    {link.label}
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">
                      {link.icon}
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-white mb-4">Connect</h3>
            <ul className="space-y-2">
              {[
                { label: 'LinkedIn', href: 'https://www.linkedin.com/in/reecerodri89/' },
                { label: 'GitHub', href: 'https://github.com/XreeceX' },
                { label: 'Email', href: 'mailto:reeceandreece2@gmail.com' },
              ].map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors flex items-center gap-2 group"
                  >
                    {link.label}
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">
                      →
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>

          <div>
            <h3 className="font-semibold text-zinc-900 dark:text-white mb-4">About</h3>
            <ul className="space-y-2">
              {[
                { label: 'My Work', href: 'https://reece-rodrigues.vercel.app/#projects' },
                { label: 'Education', href: 'https://reece-rodrigues.vercel.app/#education' },
                { label: 'Research', href: 'https://reece-rodrigues.vercel.app/#research' },
              ].map((link) => (
                <li key={link.href}>
                  <a
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-zinc-600 dark:text-zinc-400 hover:text-zinc-900 dark:hover:text-white transition-colors flex items-center gap-2 group"
                  >
                    {link.label}
                    <span className="opacity-0 group-hover:opacity-100 transition-opacity">
                      →
                    </span>
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom Footer */}
        <div className="border-t border-zinc-200 dark:border-white/10 pt-8 flex flex-col md:flex-row justify-between items-center text-sm text-zinc-600 dark:text-zinc-400">
          <p>© {currentYear} Reece Rodrigues. Crafted with intent.</p>
          <div className="flex gap-6 mt-4 md:mt-0">
            <span>London, UK</span>
            <span>•</span>
            <span>AI Engineer & Full-Stack Developer</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
