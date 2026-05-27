"""
list_slicer_tools.py — Lista todas las herramientas disponibles en MCP-Slicer.

USO:
    python list_slicer_tools.py

Requiere:
    - 3D Slicer abierto con Web Server activo (puerto 2016)
    - uvx disponible en C:\\Users\\erick\\.local\\bin\\uvx.exe
"""
import asyncio
import os

UVX_PATH = r"C:\Users\erick\.local\bin\uvx.exe"

async def main():
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    print("\n" + "="*60)
    print("MCP-Slicer — Herramientas disponibles")
    print("="*60 + "\n")

    server_params = StdioServerParameters(
        command=UVX_PATH,
        args=["mcp-slicer"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()

            tools = tools_result.tools
            print(f"Total herramientas: {len(tools)}\n")

            for i, tool in enumerate(tools, 1):
                print(f"{'─'*50}")
                print(f"[{i}] {tool.name}")
                if tool.description:
                    print(f"    Descripción: {tool.description}")
                if hasattr(tool, 'inputSchema') and tool.inputSchema:
                    schema = tool.inputSchema
                    props  = schema.get('properties', {})
                    req    = schema.get('required', [])
                    if props:
                        print(f"    Parámetros:")
                        for param, info in props.items():
                            req_str  = " (requerido)" if param in req else " (opcional)"
                            type_str = info.get('type', 'any')
                            desc_str = info.get('description', '')
                            print(f"      - {param}: {type_str}{req_str}")
                            if desc_str:
                                print(f"        {desc_str}")
                print()

    print("="*60)
    print("Para usar desde Python:")
    print('  await session.call_tool("nombre_tool", {"param": "valor"})')
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(main())