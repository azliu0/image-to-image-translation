import { useState } from "react";

import "@mantine/core/styles.css";
import { MantineProvider } from "@mantine/core";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from "react-router-dom";

import RootPage from "./routes/root";
import AboutPage from "./routes/about";
import GalleryPage from "./routes/gallery";
import NotFoundPage from "./routes/404";

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route errorElement={<NotFoundPage />}>
      <Route index path="/" element={<RootPage />} />
      <Route path="/about" element={<AboutPage />} />
      <Route path="/gallery" element={<GalleryPage />} />
    </Route>
  )
);

const App = () => {
  return (
    <MantineProvider
      withGlobalStyles
      withNormalizeCSS
      theme={{
        fontFamily:
          "-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Oxygen,Ubuntu,Cantarell,open sans,helvetica neue,sans-serif",
        headings: {
          fontFamily:
            "-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Oxygen,Ubuntu,Cantarell,open sans,helvetica neue,sans-serif",
        },
      }}
    >
      <RouterProvider router={router} />
    </MantineProvider>
  );
};

export default App;
