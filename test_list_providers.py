#!/usr/bin/env python

from provider_config import ProviderConfig

# Test our provider filtering functionality
print("Testing provider filtering for Moonshot/Kimi-K2 model\n")

# List all providers for moonshot/kimi-k2
model = "moonshot/kimi-k2"
print(f"Available providers for {model}:")
all_providers = ProviderConfig.find_providers_for_model(model, enabled_only=False)

for provider in all_providers:
    status = "Enabled" if provider.get("enabled", True) else "Disabled"
    print(f"  - {provider['name']} (ID: {provider['id']}) - {status}")
    print(f"    {provider.get('description', '')}\n")

# Get default provider
default_provider = ProviderConfig.get_default_provider_for_model(model)
if default_provider:
    print(f"Default provider for {model}: {default_provider}\n")
else:
    print(f"No default provider found for {model}\n")

# Test with a different model
model2 = "anthropic/claude-3-opus"
print(f"Available providers for {model2}:")
all_providers = ProviderConfig.find_providers_for_model(model2, enabled_only=False)

for provider in all_providers:
    status = "Enabled" if provider.get("enabled", True) else "Disabled"
    print(f"  - {provider['name']} (ID: {provider['id']}) - {status}")
    print(f"    {provider.get('description', '')}\n")

# Get default provider
default_provider = ProviderConfig.get_default_provider_for_model(model2)
if default_provider:
    print(f"Default provider for {model2}: {default_provider}\n")
else:
    print(f"No default provider found for {model2}\n")
