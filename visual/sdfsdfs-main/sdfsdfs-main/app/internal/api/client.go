package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

type Client struct {
	baseURL    string
	token      string
	httpClient *http.Client
}

func NewClient(baseURL, token string) *Client {
	return &Client{
		baseURL: baseURL,
		token:   token,
		httpClient: &http.Client{
			Timeout: 900 * time.Millisecond,
		},
	}
}

func (c *Client) GetArena(ctx context.Context) (*ArenaResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/arena", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Auth-Token", c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("arena: status %d, body: %s", resp.StatusCode, body)
	}

	var arena ArenaResponse
	if err := json.Unmarshal(body, &arena); err != nil {
		return nil, fmt.Errorf("arena: unmarshal: %w", err)
	}
	return &arena, nil
}

func (c *Client) SendCommand(ctx context.Context, cmd *CommandRequest) (*CommandResponse, error) {
	data, err := json.Marshal(cmd)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/command", bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Auth-Token", c.token)
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var cmdResp CommandResponse
	if err := json.Unmarshal(body, &cmdResp); err != nil {
		return nil, fmt.Errorf("command: status %d, body: %s", resp.StatusCode, body)
	}
	return &cmdResp, nil
}

func (c *Client) GetLogs(ctx context.Context) ([]LogMessage, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/api/logs", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("X-Auth-Token", c.token)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("logs: status %d", resp.StatusCode)
	}

	var logs []LogMessage
	if err := json.Unmarshal(body, &logs); err != nil {
		return nil, err
	}
	return logs, nil
}
