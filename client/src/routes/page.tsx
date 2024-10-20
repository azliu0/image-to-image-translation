import AnimatedPage from "../components/AnimatedPage";
import { Flex, Center, Anchor } from "@mantine/core";
import { useState, useEffect } from "react";
import classes from "./root.module.css";
import LightDarkButton from "../components/LightDarkButton";
import { MarkdownFile, parseMD } from "../utils/MarkdownUtils";
import MarkdownFormat from "../components/MarkdownFormat";
import { PageInterface } from "../App";
import detailsPage from "../md/details.md";
import galleryPage from "../md/gallery.md";
import mathPage from "../md/math.md";
import modelsPage from "../md/models.md";

const Page = ({ mdPageLink, displayTitle }: PageInterface) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [author, setAuthor] = useState<string>("");
  const [title, setTitle] = useState<string>("");
  const [date, setDate] = useState<string>("");
  const [time, setTime] = useState<string>("");
  const [detailsText, setDetailsText] = useState<string>("");

  const getAbsMDLink = (mdPageLink: string): string => {
    // mdPageLink: ../md/{name}.md
    // absMDLink: /md/{name}.md
    return mdPageLink.slice(2);
  };

  useEffect(() => {
    let page;
    if (mdPageLink === "../md/details.md") {
      page = detailsPage;
    } else if (mdPageLink === "../md/gallery.md") {
      page = galleryPage;
    } else if (mdPageLink === "../md/math.md") {
      page = mathPage;
    } else if (mdPageLink === "../md/models.md") {
      page = modelsPage;
    }
    // import(/* @vite-ignore */ mdPageLink)
    //   .then((mdPageLinkModule) => mdPageLinkModule.default)
    //   .then((mdPageFile) => fetch(mdPageFile))
    fetch(page as any)
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
            — {displayTitle}
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
                absMDLink={getAbsMDLink(mdPageLink)}
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

export default Page;
