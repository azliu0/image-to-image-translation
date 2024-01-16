import { Routes, Route, useLocation } from "react-router-dom";
import { AnimatePresence } from "framer-motion";

import RootPage from "./routes/root";
import Page from "./routes/page";
import NotFoundPage from "./routes/404";
export interface PageInterface {
  mdPageLink: string;
  displayTitle: string;
}

const pages: Array<PageInterface> = [
  { mdPageLink: "../md/details.md", displayTitle: "Details" },
  { mdPageLink: "../md/gallery.md", displayTitle: "Gallery" },
  { mdPageLink: "../md/math.md", displayTitle: "Math" },
  { mdPageLink: "../md/models.md", displayTitle: "Models" },
];

const App = () => {
  const location = useLocation();

  return (
    <>
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route index element={<RootPage />} />
          {pages.map((page, idx) => (
            <Route
              key={idx}
              path={`/${page.displayTitle.toLowerCase()}`}
              element={
                <Page
                  mdPageLink={page.mdPageLink}
                  displayTitle={page.displayTitle}
                />
              }
            />
          ))}
          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </AnimatePresence>
    </>
  );
};

export default App;
