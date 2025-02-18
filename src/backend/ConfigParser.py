import os
import sys
sys.path.append(os.path.abspath('.'))

import configparser
from src.backend.common.Distributions import UniformDistribution

class ConfigParser:
    def __init__(self, config_path: str):
        self.config = configparser.ConfigParser()
        self.config_path = config_path
        self.update()

    def update(self):
        """Reload the configuration file."""
        self.config.read(self.config_path)

    def _parse_list(self, value: str):
        """Parse a string that represents a list into an actual Python list."""
        value = value.strip()[1:-1]  # Remove the square brackets
        items = value.split(",")    # Split by commas
        parsed_items = []
        for item in items:
            item = item.strip()  # Remove extra spaces
            if item.isdigit():
                parsed_items.append(int(item))  # Convert to integer if possible
            else:
                try:
                    parsed_items.append(float(item))  # Convert to float if possible
                except ValueError:
                    parsed_items.append(item)  # Keep as string if not numeric
        return parsed_items

    def getParameters(self, filter_key=None):
        """Retrieve the configuration as a dictionary."""
        config_dict = {}
        for section in self.config.sections():
            if filter_key not in section:
                continue  # Skip keys that do not match the filter
            config_dict[section] = {}
            for key, value in self.config.items(section):
                if value.startswith("[") and value.endswith("]"):
                    # Parse the value as a list
                    parsed_values = self._parse_list(value)
                    config_dict[section][key] = parsed_values
                elif value.isdigit():
                    # Parse integer values
                    config_dict[section][key] = int(value)
                else:
                    # Parse float values or keep as string
                    try:
                        config_dict[section][key] = float(value)
                    except ValueError:
                        config_dict[section][key] = value
        return config_dict

    def writeParameters(self, parameters: dict, filter_key=None):
        """Update only the new or modified parameters."""
        self.update()  # Load existing configuration

        for section, values in parameters.items():
            if filter_key not in section:
                continue  # Skip keys that do not match the filter
            if not self.config.has_section(section):
                self.config.add_section(section)
            for key, value in values.items():
                if isinstance(value, list):
                    value = "[" + ", ".join(map(str, value)) + "]"  # Convert lists to strings
                self.config.set(section, key, str(value))

        # Write back only the updated configuration
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)


class SequenceConfig(ConfigParser):
    def getParameters(self):
        """Get parameters specific to 'Sequence' keys."""
        
        params = super().getParameters(filter_key="Sequence")
        for key, key_dict in params.items():
            if "Sequence.quantities" in key:  # Ensure `key_dict` is a dictionary
                params[key] = {
                    k: int(UniformDistribution(a=value[0], b=value[1]).random_samples())
                    if isinstance(value, list) and len(value) == 2 else value
                    for k, value in key_dict.items()
                }
        return params

    def writeParameters(self, parameters: dict):
        """Write parameters specific to 'Sequence' keys."""
        super().writeParameters(parameters, filter_key="Sequence")


class DatasetConfig(ConfigParser):
    def getParameters(self):
        """Get parameters specific to 'Dataset' keys."""
        return super().getParameters(filter_key="Dataset")

    def writeParameters(self, parameters: dict):
        """Write parameters specific to 'Dataset' keys."""
        super().writeParameters(parameters, filter_key="Dataset")
        
            
if __name__ == "__main__":
    # Example usage:
    config_path = os.path.join(os.getcwd(), 'cfg', 'config.ini')
    sequence_config = SequenceConfig(config_path)

    # Get parameters as a dictionary
    params = sequence_config.getParameters()
    print("Current Parameters:", params)

    # Modify parameters and write them back
    new_params = {
        "Dataset": {"n_sequences": 3},
        "Sequence.quantities": {"spermatozoon_n": [180, 230]}
    }
    sequence_config.writeParameters(new_params)

    # Reload and check updates
    sequence_config.update()
    updated_params = sequence_config.getParameters()
    print("Updated Parameters:", updated_params)