import * as ReactDOM from "react-dom/client";
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import { MantineProvider, createTheme } from "@mantine/core";
import {
  createBrowserRouter,
  createRoutesFromElements,
  Route,
  RouterProvider,
} from "react-router-dom";

import RootPage from "./routes/root";
import DetailsPage from "./routes/details";
import GalleryPage from "./routes/gallery";
import NotFoundPage from "./routes/404";

const router = createBrowserRouter(
  createRoutesFromElements(
    <Route errorElement={<NotFoundPage />}>
      <Route index path="/" element={<RootPage />} />
      <Route path="/details" element={<DetailsPage />} />
      <Route path="/gallery" element={<GalleryPage />} />
    </Route>
  )
);

const theme = createTheme({
  fontFamily:
    "-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Oxygen,Ubuntu,Cantarell,open sans,helvetica neue,sans-serif",
  headings: {
    fontFamily:
      "-apple-system,BlinkMacSystemFont,segoe ui,Roboto,Oxygen,Ubuntu,Cantarell,open sans,helvetica neue,sans-serif",
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <>
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <RouterProvider router={router} />
    </MantineProvider>
  </>
);
