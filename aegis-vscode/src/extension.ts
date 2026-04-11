/* -------------------------------------------------------------------------
 * Original work Copyright (c) Microsoft Corporation. All rights reserved.
 * Original work licensed under the MIT License.
 * See ThirdPartyNotices.txt in the project root for license information.
 * All modifications Copyright (c) Open Law Library. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http: // www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ----------------------------------------------------------------------- */
"use strict";

import * as net from "net";
import * as path from "path";
import * as vscode from "vscode";
import * as semver from "semver";
import * as fs from "fs";

import { PythonExtension } from "@vscode/python-extension";
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    State,
    integer,
} from "vscode-languageclient/node";
import { execFile, ExecFileException } from "child_process";
import * as JSZip from "jszip";

const MIN_PYTHON = semver.parse("3.14.0");
const MIN_BEET = semver.parse("0.113.0");

// Some other nice to haves.
// TODO: Check selected env satisfies aegis' requirements - if not offer to run the select env command.
// TODO: TCP Transport
// TODO: WS Transport
// TODO: Web Extension support (requires WASM-WASI!)

let client: LanguageClient;
let clientStarting = false;
let didClientFail = false;
let python: PythonExtension;
let logger: vscode.LogOutputChannel;

function registerCommand(
    context: vscode.ExtensionContext,
    commandIdentifier: string,
    callback: () => any,
) {
    logger.info(commandIdentifier);
    context.subscriptions.push(
        vscode.commands.registerCommand(commandIdentifier, callback),
    );
}

/**
 * This is the main entry point.
 * Called when vscode first activates the extension
 */
export async function activate(context: vscode.ExtensionContext) {
    logger = vscode.window.createOutputChannel("aegis", { log: true });
    logger.info("Extension activated.");

    await getPythonExtension();
    if (!python) {
        return;
    }

    registerCommand(context, "aegis.server.openServerLog", async () => {
        vscode.window.showTextDocument(
            vscode.Uri.file(path.join(context.extensionPath, "aegis.log")),
        );
    });

    registerCommand(context, "aegis.server.updateServer", async () => {
        const resp = await fetch(
            "https://nightly.link/TheNuclearNexus/aegis-language-server/workflows/main/main/extension.zip",
        );

        if (!resp.ok)
            return vscode.window.showErrorMessage(
                "Failed to download extension\n" + resp.statusText,
            );

        const buffer = await resp.arrayBuffer();
        const zip = await JSZip.loadAsync(buffer);

        let extension = undefined;
        for (const file in zip.files) {
            if (file.endsWith(".vsix")) {
                extension = file;
                break;
            }
        }

        if (extension === undefined) {
            return vscode.window.showErrorMessage(
                "Failed to find .vsix in zip",
            );
        }

        const filePath = path.join(context.extensionPath, "..", "update.vsix");

        fs.writeFileSync(
            filePath,
            await zip.file(extension).async("nodebuffer"),
        );

        if (context.extensionMode == vscode.ExtensionMode.Production) {
            await vscode.commands.executeCommand(
                "workbench.extensions.uninstallExtension",
                "thenuclearnexus.aegis-language-server",
            );
            vscode.commands.executeCommand(
                "workbench.extensions.installExtension",
                vscode.Uri.file(filePath),
            );
        }
    });

    // Restart language server command
    registerCommand(context, "aegis.server.restart", async () => {
        logger.info("restarting server...");
        await startLangServer(context);
    });

    // Execute command... command
    registerCommand(context, "aegis.server.executeCommand", async () => {
        await executeServerCommand();
    });

    // Restart the language server if the user switches Python envs...
    context.subscriptions.push(
        python.environments.onDidChangeActiveEnvironmentPath(async () => {
            logger.info("python env modified, restarting server...");

            await startLangServer(context);
        }),
    );

    // ... or if they change a relevant config option
    context.subscriptions.push(
        vscode.workspace.onDidChangeConfiguration(async (event) => {
            if (
                event.affectsConfiguration("aegis.server") ||
                event.affectsConfiguration("aegis.client")
            ) {
                logger.info("config modified, restarting server...");
                await startLangServer(context);
            }
        }),
    );

    context.subscriptions.push(
        vscode.workspace.onDidChangeTextDocument(async (event) => {
            const fileName = event.document.fileName;
            if (
                !fileName.match(/beet.(json|toml|ya?ml)/g) &&
                fileName != "pyproject.toml"
            )
                return;

            if (didClientFail) return;
            logger.info("Config modified, restarting server");

            await startLangServer(context);
        }),
    );

    // Start the language server once the user opens the first text document...
    context.subscriptions.push(
        vscode.workspace.onDidOpenTextDocument(async (e) => {
            if (
                (
                    await vscode.workspace.findFiles(
                        "**/*.{mcfunction,bolt}",
                        null,
                        1,
                    )
                ).length == 0
            )
                return;

            if (e.fileName.endsWith(".bolt") && e.languageId !== "mcfunction") {
                vscode.window
                    .showWarningMessage(
                        ".bolt files aren't associated with mcfunction, this might break highlighting.",
                        "Configure Language Mode",
                    )
                    .then((s) => {
                        if (s == "Configure Language Mode")
                            updateFileAssociation();
                    });
            }

            if (!client && !didClientFail) {
                await startLangServer(context);
            }
        }),
    );
}

export function deactivate(): Thenable<void> {
    return stopLangServer();
}

async function onNotification() {}

/**
 * Start (or restart) the language server.
 *
 * @param command The executable to run
 * @param args Arguments to pass to the executable
 * @param cwd The working directory in which to run the executable
 * @returns
 */
async function startLangServer(context: vscode.ExtensionContext) {
    // Don't interfere if we are already in the process of launching the server.
    if (clientStarting) {
        return;
    }

    clientStarting = true;
    if (client) {
        await stopLangServer();
    }

    didClientFail = false;

    const config = vscode.workspace.getConfiguration("aegis.server");

    const cwd = context.extensionPath;

    const serverPath = "aegis_server";

    logger.info(`cwd: '${cwd}'`);
    logger.info(`server: '${serverPath}'`);

    const resource = vscode.Uri.joinPath(vscode.Uri.file(cwd), serverPath);
    const pythonCommand = await getPythonCommand(resource);

    logger.debug("Checking enviroment...");
    const successful = await checkEnviroment(context, pythonCommand);
    if (!successful) {
        clientStarting = false;
        didClientFail = true;
        return;
    }

    const args: string[] = [];

    const sites = await getSitePackages(pythonCommand[0]);
    if (sites.length > 0) {
        args.push("--site");
        args.push(...sites.map((s) => `'${s}'`));
    }

    logger.debug(`python: ${pythonCommand.join(" ")}`);
    logger.debug(process.env.DEV);
    const serverOptions: ServerOptions =
        process.env.DEV == "true"
            ? () => {
                  return new Promise((resolve /*, reject */) => {
                      const clientSocket = new net.Socket();
                      clientSocket.connect(
                          Number.parseInt(process.env.SERVER_PORT ?? "4000"),
                          "127.0.0.1",
                          () => {
                              resolve({
                                  reader: clientSocket,
                                  writer: clientSocket,
                              });
                          },
                      );
                  });
              }
            : (() => {
                  const options = {
                      command: pythonCommand[0],
                      args: [
                          context.asAbsolutePath("language_server.pyz"),
                          ...args,
                      ],
                      options: { cwd },
                  };

                  logger.info([options.command, ...options.args].join(" "));
                  logger.info(JSON.stringify(options));

                  return options;
              })();

    client = new LanguageClient("aegis-lsp", serverOptions, getClientOptions());

    const promises = [client.start()];

    if (config.get<boolean>("debug")) {
        promises.push(startDebugging());
    }

    const results = await Promise.allSettled(promises);
    clientStarting = false;

    for (const result of results) {
        if (result.status === "rejected") {
            logger.error(
                `There was a error starting the server: ${result.reason}`,
            );
        }
    }

    logger.debug("server has started");
}

async function checkEnviroment(
    context: vscode.ExtensionContext,
    pythonCommand: string[],
): Promise<boolean> {
    const configurePythonAction = (selection: "Configure Python") => {
        if (selection != "Configure Python") return;

        if (!python)
            return vscode.window.showErrorMessage(
                "Python extension is not installed!",
            );

        vscode.commands.executeCommand("python.setInterpreter");
    };

    if (!pythonCommand) {
        vscode.window
            .showErrorMessage(
                "No Python installation configured!",
                "Configure Python",
            )
            .then(configurePythonAction);
        return false;
    }

    if (
        vscode.workspace
            .getConfiguration("aegis")
            .get("aegis.client.hide_env_warnings")
    )
        return true;

    await checkForVenv(pythonCommand[0]);

    const versionResult = await hasBeet(context, pythonCommand[0]);

    switch (versionResult) {
        case "absent":
        case "outdated":
            vscode.window
                .showErrorMessage(
                    versionResult === "absent"
                        ? "Beet is not installed in this enviroment!"
                        : `The installed version of Beet is outdated! Aegis requires atleast ${MIN_BEET.toString()}`,
                    versionResult === "absent"
                        ? "Install Beet"
                        : "Upgrade Beet",
                    "Don't Show Again",
                )
                .then(async (selection) => {
                    if (
                        selection == "Install Beet" ||
                        selection == "Upgrade Beet"
                    )
                        await installBeet(pythonCommand[0]);
                    else if (selection == "Don't Show Again") {
                        vscode.workspace
                            .getConfiguration("aegis")
                            .update(
                                "aegis.client.hide_env_warnings",
                                true,
                                vscode.ConfigurationTarget.Workspace,
                            );
                    }
                });
            return false;
        case "errored":
            vscode.window
                .showWarningMessage(
                    "An error occurred while checking for Beet. Aegis will attempt to start regardless.",
                    "Don't Show Again",
                )
                .then((selection) => {
                    if (selection == "Don't Show Again") {
                        vscode.workspace
                            .getConfiguration("aegis")
                            .update(
                                "aegis.client.hide_env_warnings",
                                true,
                                vscode.ConfigurationTarget.Workspace,
                            );
                    }
                });
            return true;
        case "present":
            return true;
    }
}

function updateFileAssociation() {
    const config = vscode.workspace.getConfiguration();
    const associations: Record<string, string> =
        config.get("files.associations");

    associations["*.bolt"] = "mcfunction";
    config.update("files.associations", associations);
}

async function runPythonCommand(
    pythonCommand: string,
    args: string[],
): Promise<[ExecFileException | null, string]> {
    return new Promise<[ExecFileException | null, string]>((resolve) => {
        execFile(pythonCommand, args, (error, stdout, _) => {
            logger.info(stdout);
            resolve([error, stdout]);
        });
    });
}

async function checkForVenv(pythonCommand: string) {
    logger.debug("Checking if python is a venv...");

    try {
        const [error, stdout] = await runPythonCommand(pythonCommand, [
            "-c",
            "import sys; print(sys.prefix != sys.base_prefix)",
        ]);

        if (error) throw error;

        if (!stdout.includes("True")) {
            vscode.window
                .showWarningMessage(
                    "It's recommended to use Beet within a Virtual Enviroment",
                    "Configure Python",
                    "Learn More",
                    "Don't Show Again",
                )
                .then((selection) => {
                    if (selection == "Configure Python") {
                        vscode.commands.executeCommand("python.setInterpreter");
                    } else if (selection == "Learn More") {
                        vscode.env.openExternal(
                            vscode.Uri.parse(
                                "https://docs.python.org/3/library/venv.html",
                            ),
                        );
                    } else if (selection == "Don't Show Again") {
                        vscode.workspace
                            .getConfiguration("aegis")
                            .update(
                                "aegis.client.hide_env_warnings",
                                true,
                                vscode.ConfigurationTarget.Workspace,
                            );
                    }
                });
        }
    } catch (e) {
        logger.error(`Error encountered while checking for venv!\n${e}`);
    }
}

async function hasBeet(
    context: vscode.ExtensionContext,
    pythonCommand: string,
): Promise<"errored" | "absent" | "outdated" | "present"> {
    logger.debug("Checking if beet is install in the enviroment...");
    try {
        const script = path.join(
            context.extensionPath,
            "resources",
            "beet_check.py",
        );
        const [error, stdout] = await runPythonCommand(pythonCommand, [script]);

        if (error) throw error;

        if (stdout.includes("False")) return "absent";

        const beetVersion = semver.parse(stdout.split(" ")[1]);

        if (beetVersion && semver.lt(beetVersion, MIN_BEET)) {
            return "outdated";
        }

        return "present";
    } catch (e) {
        logger.error(`Error encountered while checking for beet!\n${e}`);
        return "errored";
    }
}

async function installBeet(pythonCommand: string) {
    logger.debug("Installing beet and recommended packages...");
    try {
        const [error, _] = await runPythonCommand(pythonCommand, [
            "-m",
            "pip",
            "install",
            "-U",
            `beet>=${MIN_BEET.toString()}`,
            "mecha",
            "bolt",
        ]);
        if (error) throw error;

        vscode.window
            .showInformationMessage(
                "Beet has been successfully installed!",
                "Restart Language Server",
            )
            .then((selection) => {
                if (selection != "Restart Language Server") return;

                vscode.commands.executeCommand("aegis.server.restart");
            });
    } catch (e) {
        const message = `Error encountered while installing beet!\n${e}`;
        vscode.window.showErrorMessage(message.split("\n")[0]);
        logger.error(message);
        return false;
    }
}

async function getSitePackages(pythonCommand: string): Promise<string[]> {
    logger.debug("Getting python site packages...");
    try {
        const [error, json] = await runPythonCommand(pythonCommand, [
            "-c",
            "import site; import json;  print(json.dumps(site.getsitepackages()))",
        ]);

        if (error) throw error;

        logger.debug("Received", json);

        const data = JSON.parse(json);
        if (data instanceof Array) return data;
        return [];
    } catch (e) {
        logger.error(e);
        return [];
    }
}

async function stopLangServer(): Promise<void> {
    if (!client) {
        return;
    }

    if (client.state === State.Running) {
        await client.stop();
    }

    client.dispose();
    client = undefined;
}

function startDebugging(): Promise<void> {
    if (!vscode.workspace.workspaceFolders) {
        logger.error("Unable to start debugging, there is no workspace.");
        return Promise.reject(
            "Unable to start debugging, there is no workspace.",
        );
    }
    // TODO: Is there a more reliable way to ensure the debug adapter is ready?
    setTimeout(async () => {
        await vscode.debug.startDebugging(
            vscode.workspace.workspaceFolders[0],
            "aegis: Debug Server",
        );
    }, 2000);
}

function getClientOptions(): LanguageClientOptions {
    const config = vscode.workspace.getConfiguration("aegis.client");
    const options = {
        documentSelector: config.get<any>("documentSelector"),
        outputChannel: logger,
        connectionOptions: {
            maxRestartCount: 0, // don't restart on server failure.
        },
    };
    logger.info(`client options: ${JSON.stringify(options, undefined, 2)}`);
    return options;
}

/**
 * Execute a command provided by the language server.
 */
async function executeServerCommand() {
    if (!client || client.state !== State.Running) {
        await vscode.window.showErrorMessage(
            "There is no language server running.",
        );
        return;
    }

    const knownCommands =
        client.initializeResult.capabilities.executeCommandProvider?.commands;
    if (!knownCommands || knownCommands.length === 0) {
        const info = client.initializeResult.serverInfo;
        const name = info?.name || "Server";
        const version = info?.version || "";

        await vscode.window.showInformationMessage(
            `${name} ${version} does not implement any commands.`,
        );
        return;
    }

    const commandName = await vscode.window.showQuickPick(knownCommands, {
        canPickMany: false,
    });
    if (!commandName) {
        return;
    }
    logger.info(`executing command: '${commandName}'`);

    const result = await vscode.commands.executeCommand(
        commandName /* if your command accepts arguments you can pass them here */,
    );
    logger.info(
        `${commandName} result: ${JSON.stringify(result, undefined, 2)}`,
    );
}

/**
 * Return the python command to use when starting the server.
 *
 * If debugging is enabled, this will also included the arguments to required
 * to wrap the server in a debug adapter.
 *
 * @returns The full python command needed in order to start the server.
 */
async function getPythonCommand(
    resource?: vscode.Uri,
): Promise<string[] | undefined> {
    const pythonPath = await getPythonInterpreter(resource);
    if (!pythonPath) {
        return;
    }
    const command = [pythonPath];
    const enableDebugger = process.env.DEV === "true";

    if (!enableDebugger) {
        return command;
    }

    try {
        const debugArgs = await python.debug.getRemoteLauncherCommand(
            "localhost",
            4000,
            true,
        );
        // Debugpy recommends we disable frozen modules
        command.push("-Xfrozen_modules=off", ...debugArgs);
    } catch (err) {
        logger.error(`Unable to get debugger command: ${err}`);
        logger.error("Debugger will not be available.");
    }

    return command;
}

/**
 * Return the python interpreter to use when starting the server.
 *
 * This uses the official python extension to grab the user's currently
 * configured environment.
 *
 * @returns The python interpreter to use to launch the server
 */
async function getPythonInterpreter(
    resource?: vscode.Uri,
): Promise<string | undefined> {
    const config = vscode.workspace.getConfiguration("aegis.server", resource);
    const pythonPath = config.get<string>("pythonPath");
    if (pythonPath) {
        logger.info(
            `Using user configured python environment: '${pythonPath}'`,
        );
        return pythonPath;
    }

    if (!python) {
        return;
    }

    if (resource) {
        logger.info(
            `Looking for environment in which to execute: '${resource.toString()}'`,
        );
    }
    // Use whichever python interpreter the user has configured.
    const activeEnvPath =
        python.environments.getActiveEnvironmentPath(resource);
    logger.info(
        `Found environment: ${activeEnvPath.id}: ${activeEnvPath.path}`,
    );

    const activeEnv =
        await python.environments.resolveEnvironment(activeEnvPath);
    if (!activeEnv) {
        logger.error(`Unable to resolve envrionment: ${activeEnvPath}`);
        return;
    }

    const v = activeEnv.version;

    if (v.major == -1) {
        logger.error(`Could not resolve python version!`);
        return;
    }

    const pythonVersion = semver.parse(`${v.major}.${v.minor}.${v.micro}`);

    // Check to see if the environment satisfies the min Python version.
    if (semver.lt(pythonVersion, MIN_PYTHON)) {
        const message = [
            `Your currently configured environment provides Python v${pythonVersion} `,
            `but aegis requires v${MIN_PYTHON}.\n\nPlease choose another environment.`,
        ].join("");

        const response = await vscode.window.showErrorMessage(
            message,
            "Change Environment",
        );
        if (!response) {
            return;
        } else {
            await vscode.commands.executeCommand("python.setInterpreter");
            return;
        }
    }

    const pythonUri = activeEnv.executable.uri;
    if (!pythonUri) {
        logger.error(`URI of Python executable is undefined!`);
        return;
    }

    return pythonUri.fsPath;
}

async function getPythonExtension() {
    try {
        python = await PythonExtension.api();
    } catch (err) {
        logger.error(`Unable to load python extension: ${err}`);
    }
}
