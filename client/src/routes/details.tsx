import AnimatedPage from "../components/AnimatedPage";
import { Flex, Center, Anchor, Divider } from "@mantine/core";
import { useState, useEffect } from "react";
import classes from "./root.module.css";
import LightDarkButton from "../components/LightDarkButton";
import Details from "../md/details.md";
import { MarkdownFile, parseMD } from "../utils/MarkdownUtils";
import MarkdownFormat from "../components/MarkdownFormat";

const DetailsPage = () => {
  const [loading, setLoading] = useState<boolean>(true);
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
        setLoading(false);
      });
  }, []);

  return (
    <>
      <AnimatedPage>
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
            {!loading ? (
              <MarkdownFormat
                author={author}
                title={title}
                date={date}
                time={time}
                content={detailsText}
              />
            ) : (
              <div>Loading...</div>
            )}
          </Flex>
        </Center>
      </AnimatedPage>
    </>
  );
};

export default DetailsPage;
