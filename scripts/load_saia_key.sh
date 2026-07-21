#!/usr/bin/env bash

key_file="${HOME}/.config/saia/api_key"

if [[ ! -r "${key_file}" ]]; then
  printf 'SAIA key file is missing or unreadable: %s\n' "${key_file}" >&2
  return 1 2>/dev/null || exit 1
fi

export SAIA_API_KEY="$(cat "${key_file}")"

if [[ -z "${SAIA_API_KEY}" ]]; then
  printf 'SAIA key file is empty: %s\n' "${key_file}" >&2
  return 1 2>/dev/null || exit 1
fi

export SAIA_BASE_URL="${SAIA_BASE_URL:-https://chat-ai.academiccloud.de/v1}"
export SAIA_MODEL="${SAIA_MODEL:-qwen3-coder-next}"

printf 'SAIA_API_KEY loaded; base URL=%s; model=%s\n' \
  "${SAIA_BASE_URL}" "${SAIA_MODEL}"
