import * as ReactDOM from "react-dom/client";
import "@mantine/core/styles.css";
import "@mantine/dropzone/styles.css";
import { MantineProvider, createTheme } from "@mantine/core";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import App from "./App";

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
      <Router>
        <Routes>
          <Route path="*" element={<App />} />
        </Routes>
      </Router>
    </MantineProvider>
  </>
);
