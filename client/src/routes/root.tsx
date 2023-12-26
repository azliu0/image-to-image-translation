import { useState } from "react";
import LightDarkButton from "../components/LightDarkButton";
import {
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
} from "@mantine/core";
import { IconUpload, IconPhoto, IconX } from "@tabler/icons-react";
import {
  Dropzone,
  IMAGE_MIME_TYPE,
  DropzoneProps,
  FileWithPath,
} from "@mantine/dropzone";
import classes from "./root.module.css";
import { useEffect } from "react";

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

  const models: Array<ModelSelect> = [
    { value: "pix2pix-base", label: "pix2pix-base" },
    { value: "pix2pix-1", label: "pix2pix-1 (available soon)", disabled: true },
    { value: "pix2pix-2", label: "pix2pix-2 (available soon)", disabled: true },
    { value: "pix2pix-3", label: "pix2pix-3 (available soon)", disabled: true },
    { value: "pix2pix-4", label: "pix2pix-4 (available soon)", disabled: true },
    {
      value: "pix2pix-full",
      label: "pix2pix-full (available soon)",
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

  const blah = document.getElementById("blah") as HTMLImageElement;
  function readURL(files: FileWithPath[]) {
    if (files) {
      const convertedFile = files as Blob;
      blah.src = URL.createObjectURL(files);
    }
  }

  const handleMouseEnterRight = () => {
    setIsHoveredRight(true);
  };

  const handleMouseLeaveRight = () => {
    setIsHoveredRight(false);
  };

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
        </div>
        <div className={classes.subtitle}>
          Web Visualizer and Implementation of the Stable-Diffusion based
          Pix2Pix Image-to-Image Translation model
        </div>
      </Flex>
      <Center className={classes.referencesContainer}>
        <Anchor href="/details" className={classes.reference}>
          Details
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
          label="Select a model..."
          placeholder="Select a model"
          data={models}
          value={model}
          onChange={setModel}
          className={classes.modelSelection}
        />
      </Center>
      <Flex justify="center" gap={30} wrap="wrap">
        <Dropzone
          onDrop={(files) => console.log("accepted files", files)}
          onReject={(files) => console.log("rejected files", files)}
          maxSize={5 * 1024 ** 2}
          accept={IMAGE_MIME_TYPE}
          className={classes.dropzone}
          h={500}
          w={500}
        >
          <Group
            justify="center"
            gap="xl"
            style={{ pointerEvents: "none", width: 366 }}
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
              <IconPhoto
                style={{
                  width: rem(52),
                  height: rem(52),
                  color: "var(--mantine-color-dimmed)",
                }}
                stroke={1.5}
              />
            </Dropzone.Idle>

            <div>
              <img id="blah"></img>
              <Text size="xl" inline>
                Drag your starting image here or click to select an image
              </Text>
              <Text size="sm" c="dimmed" inline mt={7}>
                Attach a starting image. See the details page for guidance about
                what images work best!
              </Text>
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
            width: 500,
            height: 500,
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
        />
      </Center>
      <Center>
        <Button className={classes.genButton}>Generate</Button>
      </Center>
    </>
  );
};

export default RootPage;
