"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Icon } from "@/components/drama";
import { useWatchlist } from "@/lib/watchlist";

const LINKS = [
  { href: "/", label: "Home" },
  { href: "/explore", label: "Explore" },
  { href: "/predictor", label: "AI Predictor" },
  { href: "/insights", label: "AI Insights" },
  { href: "/watchlist", label: "My Watchlist" },
  { href: "/about", label: "About" },
];

export default function Nav() {
  const pathname = usePathname();
  const router = useRouter();
  const { items } = useWatchlist();
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => setOpen(false), [pathname]);

  const isActive = (href: string) => (href === "/" ? pathname === "/" : pathname.startsWith(href));

  const search = (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    router.push(`/explore?q=${encodeURIComponent(query.trim())}`);
    setQuery("");
    setOpen(false);
  };

  return (
    <header className={`fixed inset-x-0 top-0 z-50 transition-colors duration-500 ${scrolled || open ? "bg-ink/90 backdrop-blur-xl" : "bg-gradient-to-b from-ink/85 to-transparent"}`}>
      <nav className="mx-auto flex h-16 max-w-[1500px] items-center gap-8 px-5 sm:px-10" aria-label="Main">
        <Link href="/" className="flex shrink-0 items-baseline gap-2">
          <span className="display text-[1.7rem] tracking-wide text-paper">K-Drama</span>
          <span className="rounded-sm bg-burgundy px-1.5 py-0.5 text-[0.62rem] font-extrabold tracking-[0.14em] text-paper">AI</span>
        </Link>

        <ul className="hidden items-center gap-7 lg:flex">
          {LINKS.map((link) => {
            const active = isActive(link.href);
            return (
              <li key={link.href} className="relative">
                <Link href={link.href} aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-1.5 py-2 text-[0.82rem] font-semibold transition-colors ${active ? "text-paper" : "text-mist hover:text-paper"}`}>
                  {link.label}
                  {link.href === "/watchlist" && items.length > 0 && (
                    <span className="rounded-full bg-blush px-1.5 text-[0.62rem] font-extrabold leading-4 text-ink">{items.length}</span>
                  )}
                </Link>
                {active && <motion.span layoutId="nav-underline" className="absolute inset-x-0 -bottom-0.5 h-0.5 rounded-full bg-blush" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}
              </li>
            );
          })}
        </ul>

        <form onSubmit={search} role="search" className="relative ml-auto hidden w-56 md:block xl:w-72">
          <Icon.search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dim" />
          <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search dramas" aria-label="Search dramas"
            className="w-full rounded-md border border-line bg-ink/60 py-2 pl-9 pr-3 text-sm outline-none transition-colors placeholder:text-dim focus:border-blush" />
        </form>

        <button type="button" className="relative ml-auto h-10 w-10 md:ml-0 lg:hidden" aria-label={open ? "Close menu" : "Open menu"} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
          <span className={`absolute left-2 right-2 h-0.5 rounded bg-paper transition-transform duration-300 ${open ? "top-[19px] rotate-45" : "top-3.5"}`} />
          <span className={`absolute left-2 right-2 h-0.5 rounded bg-paper transition-transform duration-300 ${open ? "top-[19px] -rotate-45" : "top-6"}`} />
        </button>
      </nav>

      <AnimatePresence>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }} className="overflow-hidden border-t border-line px-5 lg:hidden">
            <form onSubmit={search} role="search" className="relative mt-4 md:hidden">
              <Icon.search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-dim" />
              <input type="search" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search dramas" aria-label="Search dramas" className="field !pl-9" />
            </form>
            <ul className="py-3">
              {LINKS.map((link) => (
                <li key={link.href}>
                  <Link href={link.href} className={`flex items-center justify-between py-3 text-lg font-semibold ${isActive(link.href) ? "text-blush" : "text-paper"}`}>
                    {link.label}
                    {link.href === "/watchlist" && items.length > 0 && <span className="text-sm text-mist">{items.length}</span>}
                  </Link>
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
