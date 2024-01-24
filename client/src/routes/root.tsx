import AnimatedPage from "../components/AnimatedPage";
import { useState, useEffect } from "react";
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
  NumberInput,
  Loader,
  //SimpleGrid,
} from "@mantine/core";
import { notifications } from "@mantine/notifications";
import { IconUpload, IconPhoto, IconX, IconCheck } from "@tabler/icons-react";
import { IoSettingsOutline } from "react-icons/io5";
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
  cfg: boolean;
  disabled: boolean | undefined;
}

type ModelSelect = Optional<FullModel, "disabled">;

const RootPage = () => {
  // ui states
  const [isHoveredRight, setIsHoveredRight] = useState<boolean>(false);
  const [imageSelected, setImageSelected] = useState(false);
  const [genImageSelected, setGenImageSelected] = useState(false);
  const [hiddenSettings, setHiddenSettings] = useState(true);
  const [screenWidth, setScreenWidth] = useState(window.innerWidth);
  const [loadingGeneration, setLoadingGeneration] = useState<boolean>(false);

  // model states
  const [model, setModel] = useState<string | null>(null);
  const [files, setFiles] = useState<Array<FileWithPath>>([]);
  const [genFiles, setGenFiles] = useState<Array<FileWithPath>>([]);
  const [prompt, setPrompt] = useState<string>("");
  const [inferenceSteps, setInferenceSteps] = useState<string | number>(10);
  const [temperature, setTemperature] = useState<string | number>(0.7);
  const [CFG, setCFG] = useState<string | number>(7);
  const [negativePrompt, setNegativePrompt] = useState<string | number>("");

  // error states
  const [errorModel, setErrorModel] = useState<boolean | String>(false);
  const [errorPrompt, setErrorPrompt] = useState<boolean | String>(false);

  const models: Array<ModelSelect> = [
    { value: "pix2pix-base", label: "pix2pix-base", cfg: true },
    {
      value: "pix2pix-full-no-cfg-no-ddim",
      label: "pix2pix-full-no-cfg-no-ddim",
      cfg: false,
    },
    {
      value: "pix2pix-1",
      label: "pix2pix-1 (available soon)",
      cfg: false,
      disabled: true,
    },
    {
      value: "pix2pix-2",
      label: "pix2pix-2 (available soon)",
      cfg: false,
      disabled: true,
    },
    {
      value: "pix2pix-full",
      label: "pix2pix-full (available soon)",
      cfg: true,
      disabled: true,
    },
    {
      value: "pix2pix-full-no-ddim",
      label: "pix2pix-full-no-ddim (available soon)",
      cfg: true,
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
    setFiles([files[0]]);
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

  const uploadAndFetchImage = async () => {
    setLoadingGeneration(true);
    notifications.show({
      title: "image submitted!",
      message:
        "your request is being processed. \n \
        this may take a few minutes...",
      color: "green",
      loading: false,
      icon: <IconCheck />,
      withCloseButton: true,
      autoClose: false,
    });

    const formData = new FormData();
    formData.append("model", model as string);
    formData.append("prompt", prompt);
    formData.append("inferenceSteps", inferenceSteps as string);
    formData.append("temperature", temperature as string);
    formData.append("CFG", CFG as string);
    formData.append("negativePrompt", negativePrompt as string);
    formData.append("files[]", files[0]);
    const res = await fetch(
      "https://image-to-image-translation-server.onrender.com/api/inference",
      {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "1",
        },
        body: formData,
      }
    );
    setLoadingGeneration(false);

    if (res.status < 300) {
      const data = await res.blob();
      setGenFiles([data as any]); // suppress ts(2739)
      setGenImageSelected(true);
      notifications.clean();
      notifications.show({
        title: "Success",
        message: "image finished generating!",
        color: "green",
        loading: false,
        icon: <IconCheck />,
        withCloseButton: false,
        autoClose: 2000,
      });
    } else {
      const resText = await res.json();
      notifications.clean();
      notifications.show({
        title: "inference failed",
        message: `${resText?.message || "unknown error"}`,
        color: "red",
        loading: false,
        icon: <IconX />,
        withCloseButton: true,
        autoClose: false,
      });
    }
  };

  const handleGenerate = (): void => {
    // clear output
    setGenFiles([]);
    setGenImageSelected(false);
    // clear notifications
    notifications.clean();
    // check for errors
    const hasErrors = checkErrors();
    if (!hasErrors) {
      uploadAndFetchImage();
    }
  };

  // some logic to handle screen widths for mobile
  useEffect(() => {
    // Update screenWidth whenever the window is resized
    const handleResize = () => {
      setScreenWidth(window.innerWidth);
    };

    // Add an event listener for window resize
    window.addEventListener("resize", handleResize);

    // Clean up the event listener when the component unmounts
    return () => {
      window.removeEventListener("resize", handleResize);
    };
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
          </div>
          <div className={classes.subtitle}>
            Visualizer and Implementation of the Stable-Diffusion based
            InstructPix2Pix Image-to-Image Translation model
          </div>
        </Flex>
        {screenWidth > 768 ? (
          <>
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
          </>
        ) : (
          <>
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
            </Center>
            <Center
              className={`${classes.referencesContainer} ${classes.reduceMargin}`}
            >
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
          </>
        )}
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
            <>
              {genImageSelected ? (
                // Render uploaded image if previews array has an image
                <div className={classes.previewContainer2}>
                  {previews(genFiles)}
                </div>
              ) : (
                <>
                  {loadingGeneration ? (
                    <Group className={classes.uploadBoxRightChildLoading}>
                      <div>your image is being generated!</div>
                      <div>
                        <Loader
                          color="orange"
                          size="lg"
                          className={classes.loader}
                        />
                      </div>
                      <div>
                        do not refresh the page. this might take a few minutes,
                        feel free to leave this page and come back.
                      </div>
                      <div>
                        if it is taking too long, consider lowering the number
                        of inference steps. to see how each of the settings
                        affects the quality of the final image, check out the
                        Details page!
                      </div>
                    </Group>
                  ) : (
                    <>
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
                      <Text
                        size="xl"
                        inline
                        className={classes.uploadBoxRightChildText}
                      >
                        Your generated image will appear here!
                      </Text>
                    </>
                  )}
                </>
              )}
            </>
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
        {!hiddenSettings && (
          <>
            <Center style={{ marginTop: "-2rem" }}>
              <TextInput
                label="Negative Prompt:"
                placeholder="Negative Prompt"
                className={classes.textInput}
                onChange={(e) => setNegativePrompt(e.target.value)}
              />
            </Center>

            <Flex
              justify="center"
              wrap="wrap"
              gap="1rem"
              className={classes.btnFlex}
            >
              <NumberInput
                label="Inference steps"
                value={inferenceSteps}
                onChange={setInferenceSteps}
                min={2}
                max={1000}
                allowDecimal={false}
              />
              <NumberInput
                label="Temperature"
                value={temperature}
                onChange={setTemperature}
                decimalScale={1}
                step={0.1}
                min={0.2}
                max={1.0}
              />
              <NumberInput
                label="CFG"
                value={CFG}
                onChange={setCFG}
                min={1}
                max={14}
                allowDecimal={false}
                disabled={
                  !models.find((modelType) => modelType.value === model)?.cfg
                }
              />
            </Flex>
          </>
        )}
        <div className={classes.buttons}>
          <Button
            className={classes.genButton}
            onClick={handleGenerate}
            disabled={files.length === 0}
          >
            {files.length ? "Generate" : "Upload an image to modify! "}
          </Button>
          <Button
            className={`${classes.settingsButton} ${classes.genButton}`}
            onClick={() => setHiddenSettings(!hiddenSettings)}
            leftSection={<IoSettingsOutline size={18} />}
            variant="light"
          >
            {hiddenSettings ? "Show settings" : "Close settings"}
          </Button>
        </div>
        <div className={classes.footer}>
          Made with ❤️ by{" "}
          <a href="https://azliu.cc" target="_blank">
            Andrew Liu
          </a>{" "}
          &{" "}
          <a
            href="https://www.linkedin.com/in/jack-chen-6466a71b5/"
            target="_blank"
          >
            Jack Chen
          </a>{" "}
        </div>
      </AnimatedPage>
    </>
  );
};

export default RootPage;
