import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";
// import { BlockMath, InlineMath } from "react-katex";
import "katex/dist/katex.min.css";
// import { JsxRuntimeComponents } from "react-markdown/lib";

type Props = {
  children: string;
};

const _mapProps = (props: Props) => ({
  ...props,
  remarkPlugins: [remarkMath, remarkGfm],
  rehypePlugins: [rehypeKatex, rehypeRaw],
  // components: {
  // math: ({ value }: any) => <BlockMath>{value}</BlockMath>,
  // inlineMath: ({ value }: any) => <InlineMath>{value}</InlineMath>,
  // } as Partial<JsxRuntimeComponents>,
});

const Markdown = (props: Props) => <ReactMarkdown {..._mapProps(props)} />;

export default Markdown;
