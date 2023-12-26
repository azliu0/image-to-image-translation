import { MarkdownFile } from "../utils/MarkdownUtils";
import { Flex, Anchor, Divider } from "@mantine/core";
import classes from "../routes/root.module.css";
import Markdown from "../components/MarkdownRender";
import { FaCalendarAlt } from "react-icons/fa";
import { IoIosTime } from "react-icons/io";

const MarkdownFormat = ({
  title,
  author,
  date,
  time,
  content,
}: MarkdownFile) => {
  return (
    <>
      <div className={classes.markdownTitle}>{title}</div>
      <div className={classes.markdownSubtitle}>
        <FaCalendarAlt />
        {` ${date} `}
        <IoIosTime />
        {` ${time}`}
      </div>
      <Markdown children={content} />

      <Divider />

      <Flex className={classes.footnoteContainer} direction={"column"}>
        <Anchor href={"./src/md/details.md"} className={classes.footnote}>
          Read Markdown
        </Anchor>
        <Anchor
          onClick={() => window.scroll({ top: 0, left: 0, behavior: "smooth" })}
          className={classes.footnote}
        >
          Back to Top
        </Anchor>
      </Flex>
    </>
  );
};

export default MarkdownFormat;
