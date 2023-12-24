import {
  ActionIcon,
  useMantineColorScheme,
  useComputedColorScheme,
} from "@mantine/core";
import { useState } from "react";
import { IoInvertMode } from "react-icons/io5";

function LightDarkButton() {
  const [rotation, setRotation] = useState<number>(0);
  const { setColorScheme } = useMantineColorScheme();
  const computedColorScheme = useComputedColorScheme("dark");
  const dark = computedColorScheme === "dark";

  const handleToggleMode = () => {
    setColorScheme(computedColorScheme === "light" ? "dark" : "light");
    setTimeout(() => {
      setRotation(rotation + 180);
    }, 10);
  };

  return (
    <ActionIcon
      variant="transparent"
      color={dark ? "white" : "black"}
      onClick={handleToggleMode}
      title="Toggle color scheme"
      w={36}
      h={36}
    >
      <div
        style={{
          transform: `rotate(${rotation}deg)`,
          transition: "transform 0.05s ease-in-out",
        }}
      >
        <IoInvertMode style={{ width: 30, height: 30 }} />
      </div>
    </ActionIcon>
  );
}

export default LightDarkButton;
