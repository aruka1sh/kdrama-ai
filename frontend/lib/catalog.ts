/** Types and helpers for the drama catalogue. All data comes from TMDB or TVmaze through our FastAPI backend. */

export type DramaCard = {
  /** TMDB ids are numbers, TVmaze ids look like "tvmaze-123" */
  id: number | string;
  source: string;
  name: string | null;
  original_name: string | null;
  overview: string | null;
  poster_path: string | null;
  backdrop_path: string | null;
  first_air_date: string | null;
  year: number | null;
  genres: string[];
  vote_average: number | null;
  vote_count: number;
  popularity: number | null;
};

export type Collection = { key: string; title: string; criterion: string; items: DramaCard[] };
export type HomeData = { hero: DramaCard[]; collections: Collection[] };
export type DiscoverData = { page: number; total_pages: number; total_results: number; items: DramaCard[] };

export type CastMember = {
  id: number | null;
  name: string | null;
  original_name: string | null;
  profile_path: string | null;
  character: string | null;
  episodes: number | null;
};

export type GalleryImage = { file_path: string; width: number | null; height: number | null; kind: "backdrop" | "poster"; caption: string };

export type MdlMatch = {
  title: string;
  year: number;
  genres: string;
  runtime_min: number;
  episodes: number;
  popularity_rank: number;
  public_rating: number;
};

export type DramaDetails = DramaCard & {
  tagline: string | null;
  status: string | null;
  last_air_date: string | null;
  number_of_episodes: number | null;
  number_of_seasons: number | null;
  runtime: number | null;
  networks: { name: string; logo_path: string | null }[];
  content_rating_kr: string | null;
  homepage: string | null;
  cast: CastMember[];
  trailer: { key: string; name: string | null; type: string | null } | null;
  gallery: GalleryImage[];
  posters: GalleryImage[];
  related: DramaCard[];
  related_criterion: string;
  source_url: string | null;
  mdl_match: MdlMatch | null;
};

/**
 * Image URL for a path returned by the backend.
 * TMDB gives relative paths (sizes: w185, w342, w500, w780, w1280, original).
 * TVmaze gives full URLs with two versions: "medium" (small) and "original".
 */
export function img(path: string | null | undefined, size: string) {
  if (!path) return null;
  if (path.startsWith("http")) {
    return size === "w185" ? path.replace("original_untouched", "medium_portrait") : path;
  }
  return `https://image.tmdb.org/t/p/${size}${path}`;
}

export const formatDate = (iso: string | null) =>
  iso ? new Date(iso + "T00:00:00").toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" }) : null;

export const rating = (v: number | null) => (v === null ? null : v.toFixed(1));
