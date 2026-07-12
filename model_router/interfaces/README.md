# Overview

1. Extend an interface for every model (not just client). For example, if you want claude sonnet 3.7, sonnet 4.0 and opus 4.1 models, then extend the AnthropicInterface and create an interface for each.

2. The registry enum should be extended accordingly

3. Handle model version updates individually, since we likely wanted the model version for specific capabilities.