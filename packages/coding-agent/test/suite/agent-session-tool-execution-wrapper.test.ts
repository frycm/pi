import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import type { ToolResultMessage } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, describe, expect, it } from "vitest";
import type { ExtensionAPI, ToolExecutionCall } from "../../src/core/extensions/types.ts";
import { createHarness, type Harness } from "./harness.ts";

/** An echo tool that records the arguments it ran with, and a tool that calls echo. */
function toolsExtension(received: unknown[]) {
	return (pi: ExtensionAPI): void => {
		pi.registerTool({
			name: "echo",
			label: "echo",
			description: "Echo text.",
			parameters: Type.Object({ text: Type.String() }),
			execute: async (_id, params) => {
				received.push(params);
				return { content: [{ type: "text", text: `echo: ${params.text}` }], details: {} };
			},
		});
		pi.registerTool({
			name: "relay",
			label: "relay",
			description: "Calls echo.",
			parameters: Type.Object({}),
			execute: async (_id, _params, _signal, _onUpdate, ctx) => {
				const outcome = await ctx.executeTool("echo", { text: "nested" });
				return { content: outcome.result.content, details: {} };
			},
		});
	};
}

function toolResults(harness: Harness): ToolResultMessage[] {
	return harness.session.messages.filter((message): message is ToolResultMessage => message.role === "toolResult");
}

function text(result: ToolResultMessage | undefined): string {
	return (result?.content[0] as { text?: string } | undefined)?.text ?? "";
}

describe("pi.wrapToolExecution", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	async function start(extensionFactories: ((pi: ExtensionAPI) => void)[]) {
		const harness = await createHarness({ initialActiveToolNames: [], extensionFactories });
		harnesses.push(harness);
		await harness.session.bindExtensions({});
		return harness;
	}

	it("runs after every tool_call handler, with frozen arguments the tool also receives", async () => {
		const received: unknown[] = [];
		const order: string[] = [];
		const calls: ToolExecutionCall[] = [];
		let held: Record<string, unknown> | undefined;
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				pi.on("tool_call", (event) => {
					order.push("tool_call");
					// A handler that keeps the event can still change it later; the tool must not see that.
					held = event.input as Record<string, unknown>;
				});
				pi.wrapToolExecution(async (call, next) => {
					order.push("wrap");
					calls.push(call);
					if (held) held.text = "changed after the decision";
					expect(() => {
						(call.input as Record<string, unknown>).text = "changed by the wrapper";
					}).toThrow(TypeError);
					const result = await next();
					order.push("wrapped");
					return result;
				});
			},
		]);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("echo", { text: "decided" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");

		expect(order).toEqual(["tool_call", "wrap", "wrapped"]);
		expect(received).toEqual([{ text: "decided" }]);
		expect(received[0]).toBe(calls[0]?.input);
		expect(Object.isFrozen(received[0])).toBe(true);
		expect(calls[0]).toMatchObject({ toolName: "echo", input: { text: "decided" } });
		expect(calls[0]?.sourceInfo?.path).toBeTruthy();
		expect(calls[0]?.parentToolCallId).toBeUndefined();
		expect(text(toolResults(harness)[0])).toBe("echo: decided");
	});

	it("is not reached for a call a tool_call handler blocked", async () => {
		const received: unknown[] = [];
		let wrapped = 0;
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				pi.on("tool_call", () => ({ block: true, reason: "blocked" }));
				pi.wrapToolExecution((_call, next) => {
					wrapped++;
					return next();
				});
			},
		]);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("echo", { text: "x" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");
		expect(wrapped).toBe(0);
		expect(received).toEqual([]);
	});

	it("chains in load order, and a throw refuses the call without running the tool", async () => {
		const received: unknown[] = [];
		const order: string[] = [];
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				pi.wrapToolExecution(async (_call, next) => {
					order.push("outer");
					return next();
				});
			},
			(pi) => {
				pi.wrapToolExecution(async () => {
					order.push("inner");
					throw new Error("refused at execution");
				});
			},
		]);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("echo", { text: "x" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");
		expect(order).toEqual(["outer", "inner"]);
		expect(received).toEqual([]);
		const [result] = toolResults(harness);
		expect(result?.isError).toBe(true);
		expect(text(result)).toBe("refused at execution");
	});

	it("runs the tool once even if a wrapper calls next() twice, and fails a wrapper with no result", async () => {
		const received: unknown[] = [];
		let second = "";
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				pi.wrapToolExecution(async (call, next) => {
					if (call.input.text === "empty") return undefined as never;
					const result = await next();
					second = await next().then(
						() => "ran again",
						(error: Error) => error.message,
					);
					return result;
				});
			},
		]);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("echo", { text: "twice" })], { stopReason: "toolUse" }),
			fauxAssistantMessage([fauxToolCall("echo", { text: "empty" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");
		expect(received).toEqual([{ text: "twice" }]);
		expect(second).toBe("Tool echo: next() was already called");
		const results = toolResults(harness);
		expect(results[1]?.isError).toBe(true);
		expect(text(results[1])).toContain("returned no tool result");
	});

	it("wraps calls a tool makes through ctx.executeTool, with the parent id", async () => {
		const received: unknown[] = [];
		const calls: ToolExecutionCall[] = [];
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				pi.wrapToolExecution((call, next) => {
					calls.push(call);
					return next();
				});
			},
		]);
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("relay", {})], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");
		const parent = toolResults(harness)[0]?.toolCallId;
		expect(calls.map((call) => [call.toolName, call.parentToolCallId])).toEqual([
			["relay", undefined],
			["echo", parent],
		]);
		expect(calls[1]?.toolCallId).toBe(`${parent}/1`);
		expect(received).toEqual([{ text: "nested" }]);
	});

	it("stops wrapping once the wrapper is removed, and leaves arguments unfrozen without wrappers", async () => {
		const received: unknown[] = [];
		let wrapped = 0;
		let remove: (() => void) | undefined;
		const harness = await start([
			toolsExtension(received),
			(pi) => {
				remove = pi.wrapToolExecution((_call, next) => {
					wrapped++;
					return next();
				});
			},
		]);
		remove?.();
		harness.setResponses([
			fauxAssistantMessage([fauxToolCall("echo", { text: "x" })], { stopReason: "toolUse" }),
			fauxAssistantMessage("done"),
		]);
		await harness.session.prompt("go");
		expect(wrapped).toBe(0);
		expect(received).toEqual([{ text: "x" }]);
		expect(Object.isFrozen(received[0])).toBe(false);
	});
});
