import AnimatedPage from "../components/AnimatedPage";
import { useState } from "react";
import LightDarkButton from "../components/LightDarkButton";
import {
  Image,
  Center,
  useComputedColorScheme,
  useMantineTheme,
  Button,
  Anchor,
  Flex,
  Group,
  Text,
  rem,
  Select,
  TextInput,
  MantineTheme,
  //SimpleGrid,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconUpload, IconPhoto, IconX } from "@tabler/icons-react";
import {
  Dropzone,
  IMAGE_MIME_TYPE,
  //DropzoneProps,
  FileWithPath,
} from "@mantine/dropzone";
import classes from "./root.module.css";

type Optional<T, K extends keyof T> = Pick<Partial<T>, K> & Omit<T, K>;
interface FullModel {
  value: string;
  label: string;
  disabled: boolean | undefined;
}

type ModelSelect = Optional<FullModel, "disabled">;

const RootPage = () => {
  const [model, setModel] = useState<string | null>(null);
  const [isHoveredRight, setIsHoveredRight] = useState<boolean>(false);
  const [imageSelected, setImageSelected] = useState(false);
  const [files, setFiles] = useState<Array<FileWithPath>>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [errorModel, setErrorModel] = useState<boolean | String>(false);
  const [errorPrompt, setErrorPrompt] = useState<boolean | String>(false);

  const models: Array<ModelSelect> = [
    { value: "pix2pix-base", label: "pix2pix-base" },
    {
      value: "pix2pix-full-no-cfg-no-ddim",
      label: "pix2pix-full-no-cfg-no-ddim",
    },
    { value: "pix2pix-1", label: "pix2pix-1 (available soon)", disabled: true },
    { value: "pix2pix-2", label: "pix2pix-2 (available soon)", disabled: true },
    {
      value: "pix2pix-full",
      label: "pix2pix-full (available soon)",
      disabled: true,
    },
    {
      value: "pix2pix-full-no-ddim",
      label: "pix2pix-full-no-ddim (available soon)",
      disabled: true,
    },
  ];

  const computedColorScheme = useComputedColorScheme("dark");
  const dark = computedColorScheme === "dark";

  function getHoverColor(theme: MantineTheme): string {
    if (dark) return theme.colors.dark[5];
    else return theme.colors.gray[0];
  }

  function getBorderColor(theme: MantineTheme): string {
    if (dark) return theme.colors.dark[4];
    else return theme.colors.gray[4];
  }

  function getBackgroundColor(theme: MantineTheme): string {
    if (dark) return theme.colors.dark[6];
    else return "#ffffff";
  }

  const handleMouseEnterRight = () => {
    setIsHoveredRight(true);
  };

  const handleMouseLeaveRight = () => {
    setIsHoveredRight(false);
  };

  const previews = (files: Array<FileWithPath>) => {
    const imageUrl = URL.createObjectURL(files[0]);
    return (
      <Image
        src={imageUrl}
        fit="contain"
        onLoad={() => URL.revokeObjectURL(imageUrl)}
        style={{
          maxWidth: "100%",
          maxHeight: "100%",
        }}
      />
    );
  };

  const handleDrop = (files: Array<FileWithPath>) => {
    setFiles(files);
    setImageSelected(true);
  };

  const checkErrors = (): boolean => {
    let hasErrors = false;
    if (model === null) {
      setErrorModel("select a model!");
      hasErrors = true;
    }
    if (prompt.length === 0) {
      setErrorPrompt("enter a non-empty prompt!");
      hasErrors = true;
    }
    return hasErrors;
  };

  const handleGenerate = (): void => {
    notifications.clean();
    const hasErrors = checkErrors();
    if (!hasErrors) {
      notifications.show({
        title: "uploading to server",
        message: "submitting image...",
        loading: true,
        withCloseButton: false,
        autoClose: false,
      });
    } else {
      // notifications.show({
      //   title: "submitting",
      //   message: "submitting",
      //   loading: true,
      //   withCloseButton: false,
      //   autoClose: false,
      // });
    }
    console.log(files);
  };

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
          </div>
          <div className={classes.subtitle}>
            Visualizer and Implementation of the Stable-Diffusion based
            InstructPix2Pix Image-to-Image Translation model
          </div>
        </Flex>
        <Center className={classes.referencesContainer}>
          <Anchor href="/details" className={classes.reference}>
            Details
          </Anchor>
          <p className={classes.reference}> | </p>
          <Anchor href="/math" className={classes.reference}>
            Math
          </Anchor>
          <p className={classes.reference}> | </p>
          <Anchor href="/models" className={classes.reference}>
            Models
          </Anchor>
          <p className={classes.reference}> | </p>
          <Anchor href="/gallery" className={classes.reference}>
            Gallery
          </Anchor>
          <p className={classes.reference}> | </p>
          <Anchor
            href="https://github.com/azliu0/image-to-image-translation"
            target="_blank"
            className={classes.reference}
          >
            Repo
          </Anchor>
        </Center>
        <Center>
          <Select
            label="Select a model:"
            placeholder="Select a model"
            withAsterisk
            data={models}
            value={model}
            onChange={(model) => {
              setModel(model);
              setErrorModel(false);
            }}
            className={classes.modelSelection}
            error={errorModel}
          />
        </Center>
        <Flex justify="center" gap={30} wrap="wrap">
          <Dropzone
            onDrop={handleDrop}
            onReject={(files) => console.log("rejected files", files)}
            maxSize={5 * 1024 ** 2}
            accept={IMAGE_MIME_TYPE}
            className={classes.dropzone}
            h={520}
            w={520}
          >
            <Group
              justify="center"
              gap="xl"
              style={{
                pointerEvents: "none",
                width: imageSelected ? 255 : 366,
              }}
            >
              <Dropzone.Accept>
                <IconUpload
                  style={{
                    width: rem(52),
                    height: rem(52),
                    color: "var(--mantine-color-blue-6)",
                  }}
                  stroke={1.5}
                />
              </Dropzone.Accept>
              <Dropzone.Reject>
                <IconX
                  style={{
                    width: rem(52),
                    height: rem(52),
                    color: "var(--mantine-color-red-6)",
                  }}
                  stroke={1.5}
                />
              </Dropzone.Reject>
              <Dropzone.Idle>
                {!imageSelected && (
                  <IconPhoto
                    style={{
                      width: rem(52),
                      height: rem(52),
                      color: "var(--mantine-color-dimmed)",
                    }}
                    stroke={1.5}
                  />
                )}
              </Dropzone.Idle>

              <div>
                <div>
                  {imageSelected ? (
                    // Render uploaded image if previews array has an image
                    <div className={classes.previewContainer}>
                      {previews(files)}
                    </div>
                  ) : (
                    // Render text and file input if no images are selected
                    <div>
                      {!imageSelected && (
                        <>
                          <Text size="xl" inline>
                            Drag your starting image here or click to select an
                            image
                          </Text>
                          <Text size="sm" c="dimmed" inline mt={7}>
                            Attach a starting image. See the details page for
                            guidance about what images work best!
                          </Text>
                        </>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </Group>
          </Dropzone>
          <div
            onMouseEnter={handleMouseEnterRight}
            onMouseLeave={handleMouseLeaveRight}
            className={classes.uploadBox}
            style={{
              backgroundColor: isHoveredRight
                ? getHoverColor(useMantineTheme())
                : getBackgroundColor(useMantineTheme()),
              borderColor: getBorderColor(useMantineTheme()),
              width: 520,
              height: 520,
            }}
          >
            <Center className={classes.uploadBoxRightChild}>
              <IconPhoto
                style={{
                  width: rem(52),
                  height: rem(52),
                  color: "var(--mantine-color-dimmed)",
                }}
                stroke={1.5}
              />
            </Center>
            <img id="blah"></img>
            <Text size="xl" inline>
              Your generated image will appear here!
            </Text>
          </div>
        </Flex>
        <Center>
          <TextInput
            label="Generation Prompt:"
            placeholder="Input Prompt"
            withAsterisk
            className={classes.textInput}
            error={errorPrompt}
            onChange={(e) => {
              setPrompt(e.target.value);
              setErrorPrompt(false);
            }}
          />
        </Center>
        <Center>
          <Button
            className={classes.genButton}
            onClick={handleGenerate}
            disabled={files.length === 0}
          >
            {files.length ? "Generate" : "Upload an image to modify! "}
          </Button>
        </Center>
        <div className={classes.footer}>
          Made with ❤️ by Andrew Liu & Jack Chen
        </div>
      </AnimatedPage>
    </>
  );
};

export default RootPage;
