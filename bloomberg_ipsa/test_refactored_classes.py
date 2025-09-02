#!/usr/bin/env python3
"""
Script de prueba para verificar que las clases refactorizadas funcionan correctamente
"""

from MODELO_1 import IPSALassoTracker, IPSATrackerSinRestricciones, IPSATrackerMensual

def test_lasso_tracker():
    """Probar IPSALassoTracker"""
    print("=== Probando IPSALassoTracker ===")
    try:
        tracker = IPSALassoTracker()
        tracker.load_data()
        tracker.calculate_returns()
        print("✅ IPSALassoTracker inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en IPSALassoTracker: {e}")
        return False

def test_sin_restricciones_tracker():
    """Probar IPSATrackerSinRestricciones"""
    print("\n=== Probando IPSATrackerSinRestricciones ===")
    try:
        tracker = IPSATrackerSinRestricciones()
        tracker.load_data()
        tracker.calculate_returns()
        print("✅ IPSATrackerSinRestricciones inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en IPSATrackerSinRestricciones: {e}")
        return False

def test_mensual_tracker():
    """Probar IPSATrackerMensual"""
    print("\n=== Probando IPSATrackerMensual ===")
    try:
        tracker = IPSATrackerMensual()
        tracker.load_data()
        tracker.calculate_returns()
        print("✅ IPSATrackerMensual inicializado correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en IPSATrackerMensual: {e}")
        return False

def test_inheritance():
    """Verificar que la herencia funciona correctamente"""
    print("\n=== Probando herencia de BaseIPSATracker ===")
    try:
        # Verificar que todas las clases tienen los métodos heredados
        lasso = IPSALassoTracker()
        sin_rest = IPSATrackerSinRestricciones()
        mensual = IPSATrackerMensual()
        
        # Métodos que deben estar presentes por herencia
        required_methods = [
            'load_data', 'calculate_returns', 'get_third_friday', 
            'get_fourth_week_start', 'optimize_unrestricted_weights',
            'get_summary_statistics', 'get_current_weights',
            'create_annual_returns_visualization'
        ]
        
        for method_name in required_methods:
            assert hasattr(lasso, method_name), f"IPSALassoTracker no tiene método {method_name}"
            assert hasattr(sin_rest, method_name), f"IPSATrackerSinRestricciones no tiene método {method_name}"
            assert hasattr(mensual, method_name), f"IPSATrackerMensual no tiene método {method_name}"
        
        print("✅ Todos los métodos heredados están presentes")
        return True
    except Exception as e:
        print(f"❌ Error en verificación de herencia: {e}")
        return False

def main():
    """Función principal de prueba"""
    print("🧪 Iniciando pruebas de las clases refactorizadas...\n")
    
    results = []
    results.append(test_lasso_tracker())
    results.append(test_sin_restricciones_tracker())
    results.append(test_mensual_tracker())
    results.append(test_inheritance())
    
    print(f"\n📊 Resultados: {sum(results)}/{len(results)} pruebas pasaron")
    
    if all(results):
        print("🎉 ¡Todas las pruebas pasaron! La refactorización fue exitosa.")
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar los errores arriba.")

if __name__ == "__main__":
    main()
