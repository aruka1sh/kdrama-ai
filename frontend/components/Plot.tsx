"use client";

import dynamic from "next/dynamic";

// Plotly's own types are very large; the charts here only need plain objects.
type Trace = Record<string, unknown>;
type Layout = Record<string, unknown> & { xaxis?: Record<string, unknown>; yaxis?: Record<string, unknown> };

/** Plotly is loaded only in the browser (it needs `window`). */
const Plotly = dynamic(
  async () => {
    const [{ default: createPlotlyComponent }, plotly] = await Promise.all([
      import("react-plotly.js/factory"),
      import("plotly.js-basic-dist-min"),
    ]);
    return createPlotlyComponent(plotly.default ?? plotly) as React.ComponentType<Record<string, unknown>>;
  },
  { ssr: false, loading: () => <div className="h-full w-full animate-pulse rounded-xl bg-panel-2/60" /> },
);

const AXIS = {
  gridcolor: "rgba(248,247,250,0.07)",
  zerolinecolor: "rgba(248,247,250,0.12)",
  linecolor: "rgba(248,247,250,0.15)",
  tickfont: { color: "#aeaab8", size: 11 },
  title: { font: { color: "#aeaab8", size: 12 } },
};

function mergeAxis(axis: Record<string, unknown> = {}) {
  return { ...AXIS, ...axis, title: { ...AXIS.title, ...((axis.title as object) ?? {}) } };
}

export default function Plot({ data, layout, height = 360 }: { data: Trace[]; layout?: Layout; height?: number }) {
  return (
    <div style={{ height }} className="w-full">
      <Plotly
        data={data}
        layout={{
          autosize: true,
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(0,0,0,0)",
          font: { family: "var(--font-manrope), sans-serif", color: "#f8f7fa" },
          margin: { l: 48, r: 16, t: 16, b: 44 },
          hoverlabel: { bgcolor: "#1c1c26", bordercolor: "#e8a1ba", font: { color: "#f8f7fa", size: 12 } },
          legend: { orientation: "h", y: -0.14, font: { color: "#aeaab8", size: 11 } },
          ...layout,
          xaxis: mergeAxis(layout?.xaxis),
          yaxis: mergeAxis(layout?.yaxis),
        }}
        config={{ displayModeBar: false, responsive: true }}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
