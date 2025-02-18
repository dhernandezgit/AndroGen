class Color:
    def __init__(self, r: int, g: int, b: int):
        """
        Initialize a Color object with RGB values.
        :param red: Red component (0-255)
        :param green: Green component (0-255)
        :param blue: Blue component (0-255)
        """
        self.r = self._validate_color_value(r)
        self.g = self._validate_color_value(g)
        self.b = self._validate_color_value(b)
    
    def _validate_color_value(self, value: int) -> int:
        """
        Validate that the color value is between 0 and 255.
        :param value: The color component value
        :return: Validated color value
        """
        if 0 <= value <= 255:
            return value/255.0
        raise ValueError("Color value must be between 0 and 255")
    
    def get(self):
        return [self.r, self.g, self.b]