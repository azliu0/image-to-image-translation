export interface MarkdownFile {
  title: string;
  author: string;
  date: string;
  time: string;
  content: string;
}

export const parseMD = (text: string): Promise<MarkdownFile> => {
  const rejectText =
    "Intended formatting: \n\
    ---\n\
    title: <title>\n\
    author: <author>\n\
    date: <date>\n\
    time: <time>\n\
    ---\n\
    <content>";

  const split = text.split("\n");
  if (split.length < 6) {
    return Promise.reject("Markdown file too short. " + rejectText);
  } else if (split[0] !== "---" || split[5] !== "---") {
    return Promise.reject("Markdown header not padded with ---. " + rejectText);
  } else if (!split[1].toLowerCase().startsWith("title: ")) {
    return Promise.reject(
      "Markdown header does not include title. " + rejectText
    );
  } else if (!split[2].toLowerCase().startsWith("author: ")) {
    return Promise.reject(
      "Markdown header does not include author. " + rejectText
    );
  } else if (!split[3].toLowerCase().startsWith("date: ")) {
    return Promise.reject(
      "Markdown header does not include date. " + rejectText
    );
  } else if (!split[4].toLowerCase().startsWith("time: ")) {
    return Promise.reject(
      "Markdown header does not include time. " + rejectText
    );
  }
  const title = split[1].substring(split[1].indexOf(" ") + 1);
  const author = split[2].substring(split[2].indexOf(" ") + 1);
  const date = split[3].substring(split[3].indexOf(" ") + 1);
  const time = split[4].substring(split[4].indexOf(" ") + 1);

  const content = split.slice(6).join("\n");
  return Promise.resolve({
    title,
    author,
    date,
    time,
    content,
  });
};
