import { Flex, Center, Anchor } from "@mantine/core";
import classes from "./root.module.css";
import LightDarkButton from "../components/LightDarkButton";

const GalleryPage = () => {
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
          — Gallery
        </div>
        <div className={classes.subtitle}>[work in progress...]</div>
      </Flex>
      <Center className={classes.referencesContainer}>
        <Anchor href="/" className={classes.reference2}>
          Home
        </Anchor>
        {/* <p className={classes.reference}> | </p>
        <Anchor href="/gallery" className={classes.reference}>
          Gallery
        </Anchor> */}
      </Center>
    </>
  );
};

export default GalleryPage;
