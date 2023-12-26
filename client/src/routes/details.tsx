import { Flex, Center, Anchor, Divider } from "@mantine/core";
import { useState, useEffect } from "react";
import classes from "./root.module.css";
import LightDarkButton from "../components/LightDarkButton";
import Details from "../md/details.md";
import Markdown from "../components/MarkdownRender";
import { MarkdownFile, parseMD } from "../utils/MarkdownUtils";
import { FaCalendarAlt } from "react-icons/fa";
import { IoIosTime } from "react-icons/io";

const DetailsPage = () => {
  const [author, setAuthor] = useState<string>("");
  const [title, setTitle] = useState<string>("");
  const [date, setDate] = useState<string>("");
  const [time, setTime] = useState<string>("");
  const [detailsText, setDetailsText] = useState<string>("");

  useEffect(() => {
    fetch(Details)
      .then((res) => res.text())
      .then((text: string) => parseMD(text))
      .then((parsed: MarkdownFile) => {
        setAuthor(parsed.author);
        setDate(parsed.date);
        setTime(parsed.time);
        setTitle(parsed.title);
        setDetailsText(parsed.content);
      });
  }, []);

  return (
    <>
      <Flex justify={"end"} className={classes.togglebtn}>
        <LightDarkButton />
      </Flex>

      <Flex className={classes.titleContainer}>
        <div className={classes.title}>
          <a className={classes.noUnderline} href="/">
            Image-to-Image Translation
          </a>{" "}
          — Details
        </div>
      </Flex>
      <Center className={classes.referencesContainer}>
        <Anchor href="/" className={classes.reference2}>
          Home
        </Anchor>
      </Center>
      <Center>
        <Flex direction={"column"} className={classes.markdownContainer}>
          <div className={classes.markdownTitle}>{title}</div>
          <div className={classes.markdownSubtitle}>
            <FaCalendarAlt />
            {` ${date} `}
            <IoIosTime />
            {` ${time}`}
          </div>
          <Markdown children={detailsText} />

          <Divider />

          <Flex className={classes.footnoteContainer} direction={"column"}>
            <Anchor href={"./src/md/details.md"} className={classes.footnote}>
              Read Markdown
            </Anchor>
            <Anchor
              onClick={() =>
                window.scroll({ top: 0, left: 0, behavior: "smooth" })
              }
              className={classes.footnote}
            >
              Back to Top
            </Anchor>
          </Flex>
        </Flex>
      </Center>
    </>
  );
};

export default DetailsPage;
