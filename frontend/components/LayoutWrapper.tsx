"use client";

import { usePathname } from "next/navigation";
import Navigation from "@/components/Navigation";
import QuickGenerateNavigation from "@/components/QuickGenerateNavigation";
import { useEffect, useState } from "react";

export default function LayoutWrapper({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, [pathname]);

  const isHomePage = pathname === "/";
  const isQuickGeneratePage = pathname === "/quick-generate" || pathname === "/quick-transcribe"
    || pathname === "/live-transcribe";
  const showNavigation = mounted && !isHomePage;

  // Always return consistent wrapper structure
  return (
    <div className={showNavigation ? "flex h-screen" : ""}>
      {showNavigation && (
        isQuickGeneratePage ? <QuickGenerateNavigation /> : <Navigation />
      )}
      <main className={showNavigation ? "flex-1 ml-64 overflow-auto" : ""}>
        {children}
      </main>
    </div>
  );
}
